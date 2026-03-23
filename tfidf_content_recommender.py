
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESOURCES_CSV = DATA_DIR / "resources.csv"
LEARNERS_CSV = DATA_DIR / "learners.csv"
INTERACTIONS_CSV = DATA_DIR / "train_interactions.csv"


def load_data():
    resources = pd.read_csv(RESOURCES_CSV)
    learners = pd.read_csv(LEARNERS_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)

    resources["content_text"] = (
        resources["title"].fillna("") + " " +
        resources["description"].fillna("") + " " +
        resources["topic"].fillna("") + " " +
        resources["subtopic"].fillna("") + " " +
        resources["difficulty"].fillna("") + " " +
        resources["modality"].fillna("") + " " +
        resources["tags"].fillna("")
    ).str.lower()

    return resources, learners, interactions


def build_tfidf_matrix(resources: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(resources["content_text"])
    return vectorizer, tfidf_matrix


def build_user_profile(user_id: str, resources: pd.DataFrame, interactions: pd.DataFrame, tfidf_matrix):
    user_hist = interactions[interactions["learner_id"] == user_id].copy()
    if user_hist.empty:
        raise ValueError(f"No interactions found for user {user_id}")

    merged = user_hist.merge(
        resources[["resource_id"]],
        on="resource_id",
        how="inner"
    )

    idx_map = {rid: idx for idx, rid in enumerate(resources["resource_id"])}
    resource_indices = [idx_map[rid] for rid in merged["resource_id"] if rid in idx_map]

    weights = merged["implicit_score"].values.reshape(-1, 1)
    profile = np.asarray(tfidf_matrix[resource_indices].multiply(weights).sum(axis=0))
    return profile, set(merged["resource_id"])


def recommend_top_k(user_id: str, k: int = 5):
    resources, learners, interactions = load_data()
    vectorizer, tfidf_matrix = build_tfidf_matrix(resources)
    profile, seen_resources = build_user_profile(user_id, resources, interactions, tfidf_matrix)

    sims = cosine_similarity(profile, tfidf_matrix).flatten()
    rec_df = resources.copy()
    rec_df["score"] = sims
    rec_df = rec_df[~rec_df["resource_id"].isin(seen_resources)].copy()
    rec_df = rec_df.sort_values("score", ascending=False).head(k)

    return rec_df[[
        "resource_id", "title", "topic", "subtopic", "difficulty", "modality", "score"
    ]]


def save_example_recommendations():
    resources, learners, interactions = load_data()
    sample_users = learners["learner_id"].head(5).tolist()
    all_outputs = []

    for uid in sample_users:
        recs = recommend_top_k(uid, k=5)
        recs.insert(0, "learner_id", uid)
        all_outputs.append(recs)

    final_df = pd.concat(all_outputs, ignore_index=True)
    final_df.to_csv(OUTPUT_DIR / "example_recommendations.csv", index=False)
    print("Saved:", OUTPUT_DIR / "example_recommendations.csv")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    save_example_recommendations()
