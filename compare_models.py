
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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


# ---------- Popularity baseline ----------
def popularity_scores(train: pd.DataFrame):
    pop = (
        train.groupby("resource_id")["implicit_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="popularity_score")
    )
    return pop


def popularity_recommend(user_id: str, train: pd.DataFrame, top_k: int = 5):
    seen = set(train.loc[train["learner_id"] == user_id, "resource_id"].tolist())
    pop = popularity_scores(train)
    recs = pop[~pop["resource_id"].isin(seen)].head(top_k).copy()
    return recs


# ---------- TF-IDF content model ----------
def build_tfidf(resources: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(resources["content_text"])
    return vectorizer, tfidf_matrix


def content_scores_for_user(user_id: str, resources: pd.DataFrame, train: pd.DataFrame, tfidf_matrix):
    user_hist = train[train["learner_id"] == user_id].copy()
    if user_hist.empty:
        return pd.DataFrame(columns=["resource_id", "content_score"])

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
    return df


def content_recommend(user_id: str, resources: pd.DataFrame, train: pd.DataFrame, tfidf_matrix, top_k: int = 5):
    df = content_scores_for_user(user_id, resources, train, tfidf_matrix)
    return df.sort_values("content_score", ascending=False).head(top_k).copy()


# ---------- Collaborative filtering ----------
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
        return pd.DataFrame(columns=["resource_id", "cf_score"])

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

    return pd.DataFrame({"resource_id": list(scores.keys()), "cf_score": list(scores.values())})


def cf_recommend(user_id: str, user_item_matrix: pd.DataFrame, sim_df: pd.DataFrame, top_k: int = 5):
    df = cf_scores_for_user(user_id, user_item_matrix, sim_df)
    return df.sort_values("cf_score", ascending=False).head(top_k).copy()


# ---------- Hybrid ----------
def hybrid_recommend(user_id: str, resources: pd.DataFrame, train: pd.DataFrame, tfidf_matrix,
                     user_item_matrix: pd.DataFrame, sim_df: pd.DataFrame, alpha: float = 0.6, top_k: int = 5):
    cb_df = content_scores_for_user(user_id, resources, train, tfidf_matrix)
    cf_df = cf_scores_for_user(user_id, user_item_matrix, sim_df)

    merged = pd.merge(cb_df, cf_df, on="resource_id", how="outer").fillna(0.0)
    if merged.empty:
        return merged

    merged["content_score_norm"] = min_max_normalize(merged["content_score"])
    merged["cf_score_norm"] = min_max_normalize(merged["cf_score"])
    merged["hybrid_score"] = alpha * merged["content_score_norm"] + (1 - alpha) * merged["cf_score_norm"]
    return merged.sort_values("hybrid_score", ascending=False).head(top_k).copy()


# ---------- Evaluation ----------
def evaluate_model(model_name: str, recommender_fn, test: pd.DataFrame, ks=(5, 10)):
    rows = []
    test_users = test["learner_id"].unique().tolist()

    for k in ks:
        hits = 0
        precisions, recalls, ndcgs = [], [], []

        for user_id in test_users:
            true_items = test.loc[test["learner_id"] == user_id, "resource_id"].tolist()
            recs = recommender_fn(user_id, k)
            recommended = recs["resource_id"].tolist() if not recs.empty else []

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
            "model": model_name,
            "k": k,
            "users_evaluated": len(test_users),
            "hit_rate": round(hits / len(test_users), 4) if test_users else 0,
            "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0,
            "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0,
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0,
        })

    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, outpath: Path):
    pivot = df.pivot(index="k", columns="model", values=metric)
    ax = pivot.plot(kind="bar")
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("K")
    ax.set_ylabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    resources, learners, train, test = load_data()

    _, tfidf_matrix = build_tfidf(resources)
    uim = build_user_item_matrix(train)
    sim_df = build_item_similarity(uim)

    results = []

    results.append(evaluate_model(
        "Popularity",
        lambda user_id, k: popularity_recommend(user_id, train, top_k=k),
        test
    ))

    results.append(evaluate_model(
        "Content-Based",
        lambda user_id, k: content_recommend(user_id, resources, train, tfidf_matrix, top_k=k),
        test
    ))

    results.append(evaluate_model(
        "Collaborative Filtering",
        lambda user_id, k: cf_recommend(user_id, uim, sim_df, top_k=k),
        test
    ))

    results.append(evaluate_model(
        "Hybrid",
        lambda user_id, k: hybrid_recommend(user_id, resources, train, tfidf_matrix, uim, sim_df, alpha=0.6, top_k=k),
        test
    ))

    comparison_df = pd.concat(results, ignore_index=True)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

    for metric in ["hit_rate", "precision_at_k", "recall_at_k", "ndcg_at_k"]:
        plot_metric(comparison_df, metric, OUTPUT_DIR / f"{metric}_comparison.png")

    # Example recommendations for one learner from each model
    sample_user = learners["learner_id"].iloc[0]
    pop = popularity_recommend(sample_user, train, top_k=5)
    cb = content_recommend(sample_user, resources, train, tfidf_matrix, top_k=5)
    cf = cf_recommend(sample_user, uim, sim_df, top_k=5)
    hy = hybrid_recommend(sample_user, resources, train, tfidf_matrix, uim, sim_df, alpha=0.6, top_k=5)

    example = []
    for model_name, df, score_col in [
        ("Popularity", pop, "popularity_score"),
        ("Content-Based", cb, "content_score"),
        ("Collaborative Filtering", cf, "cf_score"),
        ("Hybrid", hy, "hybrid_score"),
    ]:
        merged = df.merge(
            resources[["resource_id", "title", "topic", "subtopic", "difficulty", "modality"]],
            on="resource_id",
            how="left"
        )
        merged.insert(0, "model", model_name)
        merged.insert(1, "learner_id", sample_user)
        if score_col not in merged.columns:
            merged[score_col] = np.nan
        example.append(merged)

    example_df = pd.concat(example, ignore_index=True)
    example_df.to_csv(OUTPUT_DIR / "sample_user_model_outputs.csv", index=False)

    print("Saved:", OUTPUT_DIR / "model_comparison_metrics.csv")
    print("Saved charts for hit_rate, precision_at_k, recall_at_k, ndcg_at_k")
    print("Saved:", OUTPUT_DIR / "sample_user_model_outputs.csv")
    print("\nComparison preview:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
