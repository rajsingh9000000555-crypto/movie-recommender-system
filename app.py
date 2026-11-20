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


st.set_page_config(
    page_title="Movie Recommendation System",
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
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
    }
    .brand {
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 800;
        margin-bottom: 0;
        color: #ffffff;
        letter-spacing: -0.04em;
        text-align: center;
    }
    .subtitle {
        color: #b9bfd8;
        margin: 0.35rem 0 2rem;
        text-align: center;
        font-size: 1.05rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
        min-height: 260px;
        object-fit: cover;
    }
    .movie-title {
        min-height: 58px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.35;
    }
    div.stButton > button {
        border-radius: 999px;
        min-height: 46px;
        border: none;
        background: linear-gradient(90deg, #7c3cff, #b63fff);
        color: white;
        font-weight: 700;
    }
    div[data-testid="stSelectbox"] {
        max-width: 760px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="brand">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Find five movies that match the story, genre, and mood you love.</p>',
    unsafe_allow_html=True,
)

movies, similarity = load_movies()

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

if "recommended_movie_names" in st.session_state:
    st.subheader("Recommended for you")
    st.caption(f"Similar to {selected_movie}")

    names = st.session_state.recommended_movie_names
    posters = st.session_state.recommended_movie_posters
    columns = st.columns(len(names))
    for column, name, poster in zip(columns, names, posters):
        with column:
            with st.container(border=True):
                st.markdown(
                    f'<p class="movie-title">{name}</p>',
                    unsafe_allow_html=True,
                )
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:260px;display:flex;align-items:center;'
                        'justify-content:center;background:#24243a;border-radius:12px;'
                        'color:#b9bfd8;">'
                        'Poster unavailable</div>',
                        unsafe_allow_html=True,
                    )

