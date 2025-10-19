
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentRecommender:
    """
    A transparent, content-based recommendation engine using TF-IDF features
    over item descriptions and genres, and cosine similarity for retrieval.
    Provides human-readable explanations (top contributing terms).
    """

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df.copy()
        # Build a "content" field combining genres and description
        self.items_df["content"] = (
            self.items_df["genres"].fillna("") + " " + self.items_df["description"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        self.item_matrix = self.vectorizer.fit_transform(self.items_df["content"].values)
        # In-memory user profile vector (for interactive weighting / short-term prefs)
        self.user_profile_vec = None

    def _explain_similarity(self, vec_a, vec_b, top_k=5) -> List[Tuple[str, float]]:
        """
        Return top-k terms with highest product contribution between two TF-IDF vectors.
        """
        # Convert to dense term weights (sparse safe operations)
        a = vec_a.toarray().ravel()
        b = vec_b.toarray().ravel()
        contrib = a * b
        top_idx = np.argsort(contrib)[::-1][:top_k]
        terms = np.array(self.vectorizer.get_feature_names_out())[top_idx]
        scores = contrib[top_idx]
        return list(zip(terms.tolist(), scores.tolist()))

    def set_user_likes(self, liked_ids: List[int]):
        """
        Create a lightweight user profile as the mean of liked item vectors.
        """
        if not liked_ids:
            self.user_profile_vec = None
            return
        idx = self.items_df[self.items_df["item_id"].isin(liked_ids)].index
        sub = self.item_matrix[idx]
        self.user_profile_vec = sub.mean(axis=0)

    def recommend(self, query: str = "", liked_ids: List[int] = None, k: int = 5) -> List[Dict]:
        """
        Recommend items given an optional text query and/or list of liked item IDs.
        If both provided, we average the query vector and the user profile vector.
        """
        liked_ids = liked_ids or []
        # Build a query vector
        vecs = []
        labels = []
        if query:
            q_vec = self.vectorizer.transform([query])
            vecs.append(q_vec)
            labels.append("query")
        if liked_ids:
            self.set_user_likes(liked_ids)
            if self.user_profile_vec is not None:
                vecs.append(self.user_profile_vec)
                labels.append("profile")
        if not vecs:
            # Default: popular-first stub (here: just identity similarity -> show diverse set)
            sims = np.zeros(self.items_df.shape[0])
        else:
            # Average the vectors (normalized by count)
            avg_vec = sum(vecs) / len(vecs)
            sims = cosine_similarity(self.item_matrix, avg_vec).ravel()

        # Rank, exclude liked items to avoid echoing
        rank_idx = np.argsort(sims)[::-1]
        recs = []
        for i in rank_idx:
            item = self.items_df.iloc[i]
            if int(item["item_id"]) in liked_ids:
                continue
            score = float(sims[i])
            # Build explanation by comparing item to avg_vec
            exp_terms = self._explain_similarity(self.item_matrix[i], (sum(vecs)/len(vecs)) if vecs else self.item_matrix[i], top_k=5) if vecs else []
            recs.append({
                "item_id": int(item["item_id"]),
                "title": item["title"],
                "genres": item["genres"],
                "description": item["description"],
                "score": round(score, 4),
                "why": [{"term": t, "contribution": round(c, 4)} for t, c in exp_terms]
            })
        return recs[:k]
