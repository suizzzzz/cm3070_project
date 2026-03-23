
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
TRAIN_CSV = DATA_DIR / "train_interactions.csv"
TEST_CSV = DATA_DIR / "test_interactions.csv"


def load_data():
    resources = pd.read_csv(RESOURCES_CSV)
    learners = pd.read_csv(LEARNERS_CSV)
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    resources["content_text"] = (
        resources["title"].fillna("") + " " +
        resources["description"].fillna("") + " " +
        resources["topic"].fillna("") + " " +
        resources["subtopic"].fillna("") + " " +
        resources["difficulty"].fillna("") + " " +
        resources["modality"].fillna("") + " " +
        resources["tags"].fillna("")
    ).str.lower()

    return resources, learners, train, test


def min_max_normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    smin, smax = float(series.min()), float(series.max())
    if smax - smin == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - smin) / (smax - smin)


# ---------- Content-based component ----------
def build_tfidf(resources: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(resources["content_text"])
    return vectorizer, tfidf_matrix


def content_scores_for_user(user_id: str, resources: pd.DataFrame, train: pd.DataFrame, tfidf_matrix):
    user_hist = train[train["learner_id"] == user_id].copy()
    if user_hist.empty:
        return pd.DataFrame(columns=["resource_id", "content_score"]), set()

    idx_map = {rid: idx for idx, rid in enumerate(resources["resource_id"])}
    seen_items = set(user_hist["resource_id"].tolist())

    resource_indices = [idx_map[rid] for rid in user_hist["resource_id"] if rid in idx_map]
    weights = user_hist["implicit_score"].values.reshape(-1, 1)
    profile = np.asarray(tfidf_matrix[resource_indices].multiply(weights).sum(axis=0))
    sims = cosine_similarity(profile, tfidf_matrix).flatten()

    df = pd.DataFrame({
        "resource_id": resources["resource_id"],
        "content_score": sims
    })
    df = df[~df["resource_id"].isin(seen_items)].copy()
    return df, seen_items


# ---------- Collaborative filtering component ----------
def build_user_item_matrix(train: pd.DataFrame):
    return train.pivot_table(
        index="learner_id",
        columns="resource_id",
        values="implicit_score",
        aggfunc="mean",
        fill_value=0.0
    )


def build_item_similarity(user_item_matrix: pd.DataFrame):
    item_vectors = user_item_matrix.T
    sim = cosine_similarity(item_vectors)
    sim_df = pd.DataFrame(sim, index=item_vectors.index, columns=item_vectors.index)
    np.fill_diagonal(sim_df.values, 0.0)
    return sim_df


def cf_scores_for_user(user_id: str, user_item_matrix: pd.DataFrame, sim_df: pd.DataFrame):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["resource_id", "cf_score"]), set()

    user_vector = user_item_matrix.loc[user_id]
    seen_items = set(user_vector[user_vector > 0].index.tolist())
    candidate_items = [item for item in user_item_matrix.columns if item not in seen_items]

    scores = {}
    for candidate in candidate_items:
        sim_scores = sim_df[candidate]
        numerator = float((sim_scores * user_vector).sum())
        denominator = float(np.abs(sim_scores[user_vector > 0]).sum())
        score = numerator / denominator if denominator > 0 else 0.0
        scores[candidate] = score

    df = pd.DataFrame({"resource_id": list(scores.keys()), "cf_score": list(scores.values())})
    return df, seen_items


# ---------- Hybrid fusion ----------
def hybrid_recommend(user_id: str, alpha: float = 0.6, top_k: int = 5):
    resources, learners, train, test = load_data()
    _, tfidf_matrix = build_tfidf(resources)
    uim = build_user_item_matrix(train)
    sim_df = build_item_similarity(uim)

    cb_df, cb_seen = content_scores_for_user(user_id, resources, train, tfidf_matrix)
    cf_df, cf_seen = cf_scores_for_user(user_id, uim, sim_df)
    seen_items = cb_seen.union(cf_seen)

    merged = pd.merge(cb_df, cf_df, on="resource_id", how="outer").fillna(0.0)
    merged["content_score_norm"] = min_max_normalize(merged["content_score"])
    merged["cf_score_norm"] = min_max_normalize(merged["cf_score"])
    merged["hybrid_score"] = alpha * merged["content_score_norm"] + (1 - alpha) * merged["cf_score_norm"]

    recs = merged.sort_values("hybrid_score", ascending=False).head(top_k).copy()
    recs = recs.merge(
        resources[["resource_id", "title", "topic", "subtopic", "difficulty", "modality"]],
        on="resource_id",
        how="left"
    )
    return recs[[
        "resource_id", "title", "topic", "subtopic", "difficulty", "modality",
        "content_score", "cf_score", "hybrid_score"
    ]], seen_items


def evaluate_hybrid(alpha: float = 0.6, ks=(5, 10)):
    resources, learners, train, test = load_data()
    _, tfidf_matrix = build_tfidf(resources)
    uim = build_user_item_matrix(train)
    sim_df = build_item_similarity(uim)

    test_users = [u for u in test["learner_id"].unique()]
    rows = []

    for k in ks:
        hits = 0
        precisions, recalls, ndcgs = [], [], []

        for user_id in test_users:
            true_items = test.loc[test["learner_id"] == user_id, "resource_id"].tolist()

            cb_df, cb_seen = content_scores_for_user(user_id, resources, train, tfidf_matrix)
            cf_df, cf_seen = cf_scores_for_user(user_id, uim, sim_df)

            merged = pd.merge(cb_df, cf_df, on="resource_id", how="outer").fillna(0.0)
            if merged.empty:
                recommended = []
            else:
                merged["content_score_norm"] = min_max_normalize(merged["content_score"])
                merged["cf_score_norm"] = min_max_normalize(merged["cf_score"])
                merged["hybrid_score"] = alpha * merged["content_score_norm"] + (1 - alpha) * merged["cf_score_norm"]
                recommended = merged.sort_values("hybrid_score", ascending=False).head(k)["resource_id"].tolist()

            hit_count = sum(1 for item in true_items if item in recommended)
            hits += 1 if hit_count > 0 else 0

            precision = hit_count / k if k > 0 else 0
            recall = hit_count / len(true_items) if true_items else 0

            dcg = 0.0
            for i, item in enumerate(recommended, start=1):
                if item in true_items:
                    dcg += 1 / np.log2(i + 1)
            ideal_hits = min(len(true_items), k)
            idcg = sum(1 / np.log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits > 0 else 0
            ndcg = dcg / idcg if idcg > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        rows.append({
            "alpha": alpha,
            "k": k,
            "users_evaluated": len(test_users),
            "hit_rate": round(hits / len(test_users), 4) if test_users else 0,
            "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0,
            "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0,
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0,
        })

    return pd.DataFrame(rows)


def save_outputs(alpha: float = 0.6):
    resources, learners, train, test = load_data()

    sample_users = learners["learner_id"].head(5).tolist()
    frames = []
    for uid in sample_users:
        recs, _ = hybrid_recommend(uid, alpha=alpha, top_k=5)
        recs.insert(0, "learner_id", uid)
        frames.append(recs)

    example_df = pd.concat(frames, ignore_index=True)
    eval_df = evaluate_hybrid(alpha=alpha, ks=(5, 10))

    example_df.to_csv(OUTPUT_DIR / "example_hybrid_recommendations.csv", index=False)
    eval_df.to_csv(OUTPUT_DIR / "hybrid_evaluation_metrics.csv", index=False)

    print("Saved:", OUTPUT_DIR / "example_hybrid_recommendations.csv")
    print("Saved:", OUTPUT_DIR / "hybrid_evaluation_metrics.csv")
    print("\nEvaluation preview:")
    print(eval_df.to_string(index=False))


if __name__ == "__main__":
    save_outputs(alpha=0.6)
