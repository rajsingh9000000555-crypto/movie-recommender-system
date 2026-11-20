import ast
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).with_name("tmdb_5000_movies.csv")


def get_tmdb_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("TMDB_API_KEY")


def get_openai_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("OPENAI_API_KEY")


def _names(value):
    if not isinstance(value, str):
        return ""
    try:
        return " ".join(item["name"] for item in ast.literal_eval(value))
    except (ValueError, SyntaxError, TypeError, KeyError):
        return ""


@st.cache_data
def load_movies():
    movies = pd.read_csv(DATA_FILE)
    text_columns = ["overview", "genres", "keywords"]
    for column in text_columns:
        movies[column] = movies[column].fillna("")
    movies["tags"] = (
        movies["overview"]
        + " "
        + movies["genres"].map(_names)
        + " "
        + movies["keywords"].map(_names)
    )
    vector = TfidfVectorizer(stop_words="english")
    similarity = cosine_similarity(vector.fit_transform(movies["tags"]))
    return movies, similarity


@st.cache_data
def fetch_poster(movie_id):
    api_key = get_tmdb_api_key()
    if not api_key:
        return None
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={api_key}&language=en-US"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    poster_path = response.json().get("poster_path")
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else None


def recommend(movie, movies, similarity):
    matches = movies.index[movies["title"] == movie]
    if matches.empty:
        raise ValueError(f"Movie not found: {movie}")
    index = matches[0]
    distances = sorted(
        enumerate(similarity[index]), reverse=True, key=lambda item: item[1]
    )
    recommendations = movies.iloc[[item[0] for item in distances[1:6]]]
    names = recommendations["title"].tolist()
    posters = []
    for movie_id in recommendations["id"]:
        try:
            posters.append(fetch_poster(movie_id))
        except requests.RequestException:
            posters.append(None)
    return names, posters


def ask_chatgpt(prompt, recommendation_context):
    api_key = get_openai_api_key()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        model = st.secrets["OPENAI_MODEL"]
    except (KeyError, FileNotFoundError):
        pass

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly movie expert. Answer briefly, stay factual, "
                        "and use the provided recommendation context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Recommendation context:\n{recommendation_context}\n\nQuestion:\n{prompt}",
                },
            ],
            "temperature": 0.7,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #090b16 0%, #15162a 55%, #25163b 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .brand {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
        color: #ffffff;
    }
    .subtitle {
        color: #b9bfd8;
        margin-bottom: 1.5rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
        min-height: 260px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="brand">🎬 CineMatch</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Discover films similar to the movies you already love.</p>',
    unsafe_allow_html=True,
)

movies, similarity = load_movies()
if "recommendation_context" not in st.session_state:
    st.session_state.recommendation_context = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.container(border=True):
    selected_movie = st.selectbox(
        "Choose a movie",
        movies["title"].tolist(),
        index=0,
    )
    find = st.button("Find similar movies", type="primary", use_container_width=True)

if find:
    recommended_movie_names, recommended_movie_posters = recommend(
        selected_movie, movies, similarity
    )
    st.session_state.recommended_movie_names = recommended_movie_names
    st.session_state.recommended_movie_posters = recommended_movie_posters
    st.session_state.recommendation_context = (
        f"Selected movie: {selected_movie}\n"
        f"Recommended movies: {', '.join(recommended_movie_names)}"
    )
    st.session_state.chat_messages = []

if "recommended_movie_names" in st.session_state:
    st.subheader("Because you watched")
    st.caption(selected_movie)

    names = st.session_state.recommended_movie_names
    posters = st.session_state.recommended_movie_posters
    columns = st.columns(len(names))
    for column, name, poster in zip(columns, names, posters):
        with column:
            with st.container(border=True):
                st.markdown(f"### {name}")
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:260px;display:flex;align-items:center;'
                        'justify-content:center;background:#24243a;border-radius:12px;">'
                        'Poster unavailable</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("Ask ChatGPT")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about these recommendations"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_chatgpt(
                        prompt, st.session_state.recommendation_context
                    )
                except requests.RequestException as exc:
                    answer = f"I couldn't reach ChatGPT right now: {exc}"
                if answer is None:
                    answer = (
                        "Add OPENAI_API_KEY to Streamlit Secrets to enable "
                        "the ChatGPT assistant."
                    )
                st.markdown(answer)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

