import ast
from html import escape
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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(109, 75, 255, 0.22), transparent 28rem),
            radial-gradient(circle at 92% 10%, rgba(32, 196, 184, 0.12), transparent 24rem),
            #080b14;
        color: #f7f8fc;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1240px;
        padding: 3.5rem 2rem 4rem;
    }
    .eyebrow {
        color: #8e9cff;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        margin: 0 0 0.8rem;
        text-transform: uppercase;
    }
    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.5rem, 6vw, 4.7rem);
        font-weight: 700;
        line-height: 1.02;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.04em;
    }
    .subtitle {
        color: #a7aec6;
        font-size: 1.08rem;
        line-height: 1.6;
        margin: 1rem 0 2rem;
        max-width: 620px;
    }
    .hero-rule {
        background: linear-gradient(90deg, #6573ff, transparent);
        border: 0;
        height: 2px;
        margin: 0 0 2rem;
        opacity: 0.8;
        width: 130px;
    }
    .stat-label {
        color: #828ba7;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .stat-value {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(18, 23, 39, 0.78);
        border: 1px solid rgba(164, 174, 218, 0.16);
        border-radius: 18px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    }
    .control-heading {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0 0 0.2rem;
    }
    .control-copy {
        color: #8f98b4;
        font-size: 0.9rem;
        margin: 0 0 1rem;
    }
    div[data-testid="stSelectbox"] label {
        color: #b9c0d8;
        font-size: 0.82rem;
        font-weight: 600;
    }
    div[data-baseweb="select"] > div {
        background: #0e1322;
        border-color: #303957;
        border-radius: 10px;
        min-height: 46px;
    }
    .movie-title {
        min-height: 62px;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.35;
    }
    .movie-rank {
        color: #7e8cff;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .movie-meta {
        color: #cdd2e8;
        min-height: 106px;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .movie-meta strong {
        color: #ffffff;
    }
    div.stButton > button {
        border-radius: 999px;
        min-height: 48px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(100deg, #5865f2, #8b5cf6);
        color: white;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 10px 24px rgba(88, 101, 242, 0.28);
        transform: translateY(-1px);
    }
    h2 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }
    [data-testid="stCaptionContainer"] {
        color: #8f98b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="eyebrow">PERSONALIZED DISCOVERY</p>', unsafe_allow_html=True)
st.markdown('<p class="brand">Find your next<br>favorite film.</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">A smarter way to explore cinema. Choose a film you love and get five recommendations matched by story, genre, and mood.</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="hero-rule">', unsafe_allow_html=True)

movies, similarity = load_movies()

stats = st.columns(3)
with stats[0]:
    st.markdown(
        f'<div class="stat-label">Library</div><div class="stat-value">{len(movies):,} films</div>',
        unsafe_allow_html=True,
    )
with stats[1]:
    st.markdown(
        '<div class="stat-label">Method</div><div class="stat-value">Content matched</div>',
        unsafe_allow_html=True,
    )
with stats[2]:
    st.markdown(
        '<div class="stat-label">Results</div><div class="stat-value">5 at a time</div>',
        unsafe_allow_html=True,
    )

st.write("")
with st.container(border=True):
    st.markdown('<p class="control-heading">Start with a movie you love</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="control-copy">We will surface films with a similar cinematic fingerprint.</p>',
        unsafe_allow_html=True,
    )
    selected_movie = st.selectbox(
        "Your reference movie",
        movies["title"].tolist(),
        index=0,
    )
    find = st.button("Discover recommendations", type="primary", use_container_width=True)

if find:
    recommended_movie_names, recommended_movie_details = recommend(
        selected_movie, movies, similarity
    )
    st.session_state.recommended_movie_names = recommended_movie_names
    st.session_state.recommended_movie_details = recommended_movie_details

if "recommended_movie_names" in st.session_state:
    st.subheader("Your next watchlist")
    st.caption(f"Similar to {selected_movie}")

    names = st.session_state.recommended_movie_names
    details = st.session_state.recommended_movie_details
    columns = st.columns(len(names))
    for rank, (column, name, detail) in enumerate(zip(columns, names, details), start=1):
        with column:
            with st.container(border=True):
                st.markdown(
                    f'<div class="movie-rank">MATCH 0{rank}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="movie-title">{escape(name)}</p>',
                    unsafe_allow_html=True,
                )
                genres, cast, release_year = detail
                genre_text = ", ".join(genres[:3]) if genres else "Not available"
                cast_text = ", ".join(cast) if cast else "Not available"
                year_text = release_year or "Not available"
                st.markdown(
                    f'<div class="movie-meta">'
                    f'<strong>{escape(year_text)}</strong> &nbsp;·&nbsp; {escape(genre_text)}<br><br>'
                    f'<strong>Featuring</strong><br>{escape(cast_text)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

