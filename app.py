import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).with_name("tmdb_5000_movies.csv")
CREDITS_FILE = Path(__file__).with_name("tmdb_5000_credits.csv")


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

    if CREDITS_FILE.exists():
        credits = pd.read_csv(CREDITS_FILE)
        movies = movies.merge(
            credits.rename(columns={"movie_id": "credits_movie_id"}),
            on="title",
            how="left",
        )
    else:
        movies["cast"] = ""

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


def parse_names(value, limit=None):
    if not isinstance(value, str) or not value:
        return []
    try:
        names = [
            item["name"]
            for item in ast.literal_eval(value)
            if isinstance(item, dict) and item.get("name")
        ]
    except (ValueError, SyntaxError, TypeError):
        return []
    return names if limit is None else names[:limit]


def get_movie_details(movie):
    genres = parse_names(movie["genres"])
    cast = parse_names(movie.get("cast"), limit=4)

    release_date = movie.get("release_date")
    release_year = ""
    if isinstance(release_date, str) and len(release_date) >= 4:
        release_year = release_date[:4]

    return genres, cast, release_year


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
    details = [get_movie_details(movie) for _, movie in recommendations.iterrows()]
    return names, details


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
    .movie-title {
        min-height: 58px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.35;
    }
    .movie-meta {
        color: #cdd2e8;
        min-height: 92px;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .movie-meta strong {
        color: #ffffff;
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
    recommended_movie_names, recommended_movie_details = recommend(
        selected_movie, movies, similarity
    )
    st.session_state.recommended_movie_names = recommended_movie_names
    st.session_state.recommended_movie_details = recommended_movie_details

if "recommended_movie_names" in st.session_state:
    st.subheader("Recommended for you")
    st.caption(f"Similar to {selected_movie}")

    names = st.session_state.recommended_movie_names
    details = st.session_state.recommended_movie_details
    columns = st.columns(len(names))
    for column, name, detail in zip(columns, names, details):
        with column:
            with st.container(border=True):
                st.markdown(
                    f'<p class="movie-title">{name}</p>',
                    unsafe_allow_html=True,
                )
                genres, cast, release_year = detail
                genre_text = ", ".join(genres[:3]) if genres else "Not available"
                cast_text = ", ".join(cast) if cast else "Not available"
                year_text = release_year or "Not available"
                st.markdown(
                    f'<div class="movie-meta">'
                    f'<strong>Year</strong><br>{year_text}<br>'
                    f'<strong>Genres</strong><br>{genre_text}<br>'
                    f'<strong>Cast</strong><br>{cast_text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

