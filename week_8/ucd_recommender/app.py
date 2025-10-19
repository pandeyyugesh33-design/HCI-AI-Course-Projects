
import streamlit as st
import pandas as pd
from recommender import ContentRecommender

st.set_page_config(page_title="UCD Recommender Demo", page_icon="🎯", layout="wide")

# --- Header
st.title("🎯 Simple AI Recommender (Content-Based)")
st.caption("Built with TF-IDF + cosine similarity. Designed with usability heuristics in mind.")

# Load data and init model (cache for performance & visibility of system status)
@st.cache_resource
def load_model():
    items = pd.read_csv("data/items.csv")
    model = ContentRecommender(items)
    return items, model

items, model = load_model()

with st.sidebar:
    st.header("Your Preferences")
    liked = st.multiselect(
        "Pick items you like (to personalize recommendations):",
        options=items["title"].tolist(),
    )
    liked_ids = items[items["title"].isin(liked)]["item_id"].tolist()
    st.markdown("**Controls**")
    reset = st.button("Reset selections")
    st.markdown("---")
    st.subheader("About")
    st.write(
        "This demo adapts ideas from content‑based recommenders such as "
        "[Ibtesam's Kaggle example](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system) "
        "and [Rounak Banik's notebook](https://www.kaggle.com/code/rounakbanik/movie-recommender-systems), "
        "but extends them with an explainability panel and a lightweight user profile."
    )
    st.write("Attribution and license in README.md.")

if reset:
    st.experimental_rerun()

st.subheader("Search (optional)")
query = st.text_input("Describe what you want (e.g., 'funny cooking romance', 'quantum heist action')")

k = st.slider("How many results?", min_value=3, max_value=10, value=5)

with st.spinner("Computing recommendations..."):
    recs = model.recommend(query=query, liked_ids=liked_ids, k=k)

st.success(f"Found {len(recs)} recommendation(s).")

# Results
for rec in recs:
    with st.container(border=True):
        st.markdown(f"### {rec['title']}  \n*{rec['genres']}* — score: **{rec['score']}**")
        st.write(rec["description"])
        if rec["why"]:
            st.markdown("**Why this is recommended**")
            st.caption("Top contributing terms from your query/preferences")
            st.table(pd.DataFrame(rec["why"]))

# Heuristics aids: help, undo, and feedback
with st.expander("♿ Accessibility & Help"):
    st.write(
        "Keyboard friendly, clear labels, simple language. You can remove items from 'Your Preferences' to update suggestions."
    )

st.markdown("---")
st.markdown("**Notes on usability heuristics applied:** Visibility of system status (spinner & success), match to real world (plain language), user control (Reset), consistency and standards (Streamlit UI patterns), error prevention (multiselects/constraints), recognition over recall (lists), minimalist design, and help.")
