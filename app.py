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


st.header("Movie Recommender System")
movies, similarity = load_movies()
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown", movies["title"].tolist()
)

if st.button("Show Recommendation"):
    recommended_movie_names, recommended_movie_posters = recommend(
        selected_movie, movies, similarity
    )
    columns = st.columns(len(recommended_movie_names))
    for column, name, poster in zip(
        columns, recommended_movie_names, recommended_movie_posters
    ):
        with column:
            st.text(name)
            if poster:
                st.image(poster)

