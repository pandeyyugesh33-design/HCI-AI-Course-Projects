
# UCD Recommender Demo (Streamlit)

A simple, transparent content-based recommendation demo designed with user-centered design (UCD) principles and Nielsen's usability heuristics in mind.

## How it works
- Builds a TF‑IDF representation over item *genres + description*.
- Uses cosine similarity to retrieve the most similar items to your text query and/or an on-the-fly user profile (average of liked items).
- Shows **explanations** (top contributing terms) for each recommendation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Attribution
This project adapts and extends ideas from the following examples:

- **Ibtesam.** (Kaggle) *Getting Started with a Movie Recommendation System*. Content-based filtering using TF-IDF and cosine similarity. https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system
- **Rounak Banik.** (Kaggle) *Movie Recommender Systems*. A set of simple recommenders including content-based approaches. https://www.kaggle.com/code/rounakbanik/movie-recommender-systems

**Changes & extensions in this repo:**
- Consolidated a transparent content representation (*genres + description* n‑grams).
- Added an interactive **Streamlit GUI** with clear controls and spinner/status.
- Added a **user profile vector** from selected likes to combine with queries.
- Added **explainability** panel with top contributing terms per item.
- Applied **usability heuristics** (visibility of status, user control, consistency, etc.).

## License
MIT. See `LICENSE`.
