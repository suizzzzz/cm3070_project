
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_interactions.csv"
TEST_CSV = DATA_DIR / "test_interactions.csv"
RESOURCES_CSV = DATA_DIR / "resources.csv"
LEARNERS_CSV = DATA_DIR / "learners.csv"


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    resources = pd.read_csv(RESOURCES_CSV)
    learners = pd.read_csv(LEARNERS_CSV)
    return train, test, resources, learners


def build_user_item_matrix(train: pd.DataFrame):
    matrix = train.pivot_table(
        index="learner_id",
        columns="resource_id",
        values="implicit_score",
        aggfunc="mean",
        fill_value=0.0
    )
    return matrix


def compute_item_similarity(user_item_matrix: pd.DataFrame):
    item_vectors = user_item_matrix.T
    sim = cosine_similarity(item_vectors)
    sim_df = pd.DataFrame(sim, index=item_vectors.index, columns=item_vectors.index)
    np.fill_diagonal(sim_df.values, 0.0)
    return sim_df


def recommend_item_based_cf(user_id: str, user_item_matrix: pd.DataFrame, sim_df: pd.DataFrame, top_k: int = 5):
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in training matrix")

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

    recs = pd.DataFrame({
        "resource_id": list(scores.keys()),
        "cf_score": list(scores.values())
    }).sort_values("cf_score", ascending=False).head(top_k)

    return recs, seen_items


def evaluate_leave_one_out(user_item_matrix: pd.DataFrame, sim_df: pd.DataFrame, test: pd.DataFrame, ks=(5, 10)):
    results = []
    test_users = [u for u in test["learner_id"].unique() if u in user_item_matrix.index]

    for k in ks:
        hits = 0
        precisions, recalls, ndcgs = [], [], []

        for user_id in test_users:
            true_items = test.loc[test["learner_id"] == user_id, "resource_id"].tolist()
            recs, _ = recommend_item_based_cf(user_id, user_item_matrix, sim_df, top_k=k)
            recommended = recs["resource_id"].tolist()

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

        results.append({
            "k": k,
            "users_evaluated": len(test_users),
            "hit_rate": round(hits / len(test_users), 4) if test_users else 0,
            "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0,
            "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0,
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0,
        })

    return pd.DataFrame(results)


def save_outputs():
    train, test, resources, learners = load_data()
    uim = build_user_item_matrix(train)
    sim_df = compute_item_similarity(uim)

    sample_users = learners["learner_id"].head(5).tolist()
    example_frames = []
    for uid in sample_users:
        if uid in uim.index:
            recs, seen_items = recommend_item_based_cf(uid, uim, sim_df, top_k=5)
            recs = recs.merge(resources[["resource_id", "title", "topic", "subtopic", "difficulty", "modality"]], on="resource_id", how="left")
            recs.insert(0, "learner_id", uid)
            example_frames.append(recs[["learner_id", "resource_id", "title", "topic", "subtopic", "difficulty", "modality", "cf_score"]])

    if example_frames:
        example_df = pd.concat(example_frames, ignore_index=True)
    else:
        example_df = pd.DataFrame()

    eval_df = evaluate_leave_one_out(uim, sim_df, test, ks=(5, 10))

    example_df.to_csv(OUTPUT_DIR / "example_cf_recommendations.csv", index=False)
    eval_df.to_csv(OUTPUT_DIR / "cf_evaluation_metrics.csv", index=False)

    print("Saved:", OUTPUT_DIR / "example_cf_recommendations.csv")
    print("Saved:", OUTPUT_DIR / "cf_evaluation_metrics.csv")
    print("\nEvaluation preview:")
    print(eval_df.to_string(index=False))


if __name__ == "__main__":
    save_outputs()
