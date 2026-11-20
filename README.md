# Movie Recommender System

A content-based movie recommender using the included TMDB movie dataset and cosine
similarity.

## Run the app

From this directory, install the dependencies and start Streamlit:

```bash
pip install pandas requests scikit-learn streamlit
streamlit run app.py
```

The app reads `tmdb_5000_movies.csv` directly, so generated pickle files are not
required. Poster images use the TMDB API and are skipped if the API is unavailable.

The notebook can be run locally from this directory. The credits CSV is optional;
without it, recommendations use overview, genre, and keyword data.
