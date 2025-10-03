import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from data_utils import stratified_splits, save_report_and_cm


def main():
    # CLI args for tuning
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/IMDB_clean.csv",
                    help="Path to CSV.")
    ap.add_argument("--val_size", type=float, default=0.1, help="Validation size (fraction of train).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test size (fraction of all data).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits and sampling.")
    ap.add_argument("--max_features", type=int, default=100000, help="Max TF-IDF features.")
    ap.add_argument("--ngrams", type=int, default=2, help="Use up to n-grams (1..n).")
    ap.add_argument("--n_jobs", type=int, default=-1, help="Parallelism for GridSearchCV.")
    args = ap.parse_args()

    # Output folders (artifacts)
    out_models = Path("models")
    out_fig = Path("reports/figures")
    out_metrics = Path("reports/metrics")
    for p in (out_models, out_fig, out_metrics):
        p.mkdir(parents=True, exist_ok=True)

    # Load the cleaned the dataset
    DATA_PATH = Path(args.data_path)
    df = pd.read_csv(DATA_PATH)

    # Stratified train/val/test
    train_df, val_df, test_df = stratified_splits(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    X_train, Y_train = train_df['review'].tolist(), train_df["sentiment"].tolist()
    X_val, Y_val= val_df['review'].tolist(), val_df["sentiment"].tolist()
    X_test, Y_test = test_df['review'].tolist(), test_df["sentiment"].tolist()

    # Feature extractor (shared across all models)
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, args.ngrams),
        max_features=args.max_features,
    )


    # Models & simple hypergrid
    models = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=4000, solver="saga"),
        "svm": LinearSVC(),
    }
    grids = {
        "nb": {"clf__alpha": [0.05, 0.1]},
        "lr": {"clf__C": [0.3, 1.0, 2.0]},
        "svm": {"clf__C": [0.3, 1.0, 2.0]},
    }

    # Track validation results and per-model test summaries
    val_results = []
    summary_rows = []

    # Also maintain a cumulative summary across multiple runs
    summary_all_path = out_metrics / "summary_all_models.csv"
    try:
        summary_all = pd.read_csv(summary_all_path)
    except Exception:
        summary_all = pd.DataFrame()

    best_name, best_f1 = None, -1.0

    # Train/Evaluate each model family (NB, LR, LinearSVC)
    for name, clf in models.items():
        # Build a pipeline: preprocessor -> classifier
        pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])

        # Small grid search to pick a reasonable setting by macro-F1
        gs = GridSearchCV(
            pipe,
            grids[name],
            scoring="f1_macro",
            cv=3,
            n_jobs=args.n_jobs,
            verbose=1,
        )
        gs.fit(X_train, Y_train)

        # Keep a simple validation check on the val set
        y_val_pred = gs.predict(X_val)
        f1_val = f1_score(Y_val, y_val_pred, average="macro")
        val_results.append(
            {"model": name, "best_params": gs.best_params_, "val_f1_macro": f1_val}
        )

        # Track the best validation model 
        if f1_val > best_f1:
            best_f1 = f1_val
            best_name = name

        # Test evaluation & saves
        best_estimator = gs.best_estimator_
        y_test_pred = best_estimator.predict(X_test)

        # Save a joblib of the fitted pipeline for later use
        joblib.dump(best_estimator, out_models / f"{name}_tfidf_model.joblib")

        # Save reports, confusion matrix, and a compact summary row
        row = save_report_and_cm(Y_test, y_test_pred, f"{name}_tfidf", out_metrics, out_fig)
        summary_rows.append(row)

        # Save misclassified rows to help with error analysis
        (pd.DataFrame({"text": X_test, "true": Y_test, "pred": y_test_pred})
         .query("true != pred")
         .to_csv(out_metrics / f"{name}_tfidf_misclassified.csv", index=False))

    # Add validation & summary data
    with open(out_metrics / "tfidf_grid_val_results.json", "w") as f:
        json.dump(val_results, f, indent=2)

    # Per-run summary for just the TF-IDF baselines in THIS run
    pd.DataFrame(summary_rows).to_csv(out_metrics / "summary_tfidf_models.csv", index=False)

    # Append to the cumulative summary (across all runs / options)
    summary_all = pd.concat([summary_all, pd.DataFrame(summary_rows)], ignore_index=True)
    summary_all.to_csv(summary_all_path, index=False)

    # Console display
    print("\nValidation results (macro-F1):")
    for r in val_results:
        print(f" - {r['model']}: {r['val_f1_macro']:.4f}  (best_params={r['best_params']})")
    print(f"\nBest validation model this run: {best_name} (macro-F1={best_f1:.4f})")

if __name__ == "__main__":
    main()
