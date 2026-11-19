"""Model training, selection, calibration, decision policy, test evaluation and packaging.

Usage:
    python -m fraud.train            # needs data/processed/ from `python -m fraud.data`

Steps (each mirrors a section of notebooks/fraud_detection_end_to_end.ipynb):
    1. Shortlist six model families on the validation block.
    2. Tune gradient boosting with time-series cross-validation.
    3. Select the best model by validation PR-AUC; paired bootstrap against the runner-up.
    4. Platt-calibrate on validation.
    5. Choose the cheapest decision policy on validation (euro cost).
    6. Evaluate ONCE on the untouched test block with bootstrap confidence intervals.
    7. Save the decision-function bundle, metrics, model card and figures.
"""
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud.config import load_config
from fraud.data import TARGET, load_splits
from fraud.evaluation import (POLICIES, apply_policy, decision_cost, policy_report, rank_metrics, scores_of,
                              stratified_bootstrap_idx)
from fraud.features import OUT_COLS, RAW_COLS, FraudFeatures
from fraud.monitoring import make_reference


def log(msg):
    print(f"[train] {msg}", flush=True)


def shortlist_models(seed: int) -> dict:
    return {
        "Dummy (prior)": make_pipeline(FraudFeatures(), DummyClassifier(strategy="prior")),
        "Logistic regression": make_pipeline(FraudFeatures(), StandardScaler(),
                                             LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)),
        "Random Forest": make_pipeline(FraudFeatures(),
                                       RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                                                              class_weight="balanced_subsample", n_jobs=-1,
                                                              random_state=seed)),
        "Gradient boosting (HGB)": make_pipeline(FraudFeatures(),
                                                 HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                                random_state=seed)),
        "MLP": make_pipeline(FraudFeatures(), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                           max_iter=200, random_state=seed)),
        "Isolation Forest (unsupervised)": make_pipeline(FraudFeatures(),
                                                         IsolationForest(n_estimators=300, random_state=seed,
                                                                         n_jobs=-1)),
    }


def tune_hgb(X, y, cfg, seed):
    param_dist = {
        "histgradientboostingclassifier__learning_rate": [0.02, 0.05, 0.1],
        "histgradientboostingclassifier__max_leaf_nodes": [15, 31, 63],
        "histgradientboostingclassifier__min_samples_leaf": [20, 50, 100],
        "histgradientboostingclassifier__l2_regularization": [0.0, 0.1, 1.0],
        "histgradientboostingclassifier__max_iter": [200, 400],
        "histgradientboostingclassifier__class_weight": [None, "balanced"],
    }
    base = make_pipeline(FraudFeatures(), HistGradientBoostingClassifier(early_stopping=False, random_state=seed))
    search = RandomizedSearchCV(base, param_dist, n_iter=cfg["tuning"]["n_iter"], scoring="average_precision",
                                cv=TimeSeriesSplit(n_splits=cfg["tuning"]["cv_splits"]), n_jobs=-1,
                                random_state=seed, refit=True)
    search.fit(X, y)
    return search


def main():
    cfg = load_config()
    seed, c_fp = cfg["seed"], float(cfg["costs"]["false_positive_eur"])
    np.random.seed(seed)
    art, figs = Path(cfg["artifacts_dir"]), Path(cfg["figures_dir"])
    art.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.dpi"] = 110
    t_start = time.time()

    train, val, test = load_splits(cfg)
    X_train, y_train = train[RAW_COLS], train[TARGET].values
    X_val, y_val = val[RAW_COLS], val[TARGET].values
    X_test, y_test = test[RAW_COLS], test[TARGET].values
    amt_val, amt_test = X_val["Amount"].values, X_test["Amount"].values
    log(f"train {len(train):,} rows/{y_train.sum()} frauds | validation {len(val):,}/{y_val.sum()} | "
        f"test {len(test):,}/{y_test.sum()}")

    # 1. Shortlist ------------------------------------------------------------------------------------
    fitted, val_scores, rows = {}, {}, []
    for name, model in shortlist_models(seed).items():
        t0 = time.time()
        model.fit(X_train) if "Isolation" in name else model.fit(X_train, y_train)
        secs = time.time() - t0
        s = scores_of(model, X_val)
        fitted[name], val_scores[name] = model, s
        rows.append({"model": name, **rank_metrics(y_val, s), "fit seconds": round(secs, 1)})
        log(f"{name:34s} validation PR-AUC={rows[-1]['PR-AUC']:.3f} ({secs:.0f}s)")

    # 2. Tune gradient boosting ------------------------------------------------------------------------
    log("tuning gradient boosting (random search, time-series CV) ...")
    t0 = time.time()
    search = tune_hgb(X_train, y_train, cfg, seed)
    best_params = {k.split("__")[1]: v for k, v in search.best_params_.items()}
    log(f"tuning done in {time.time() - t0:.0f}s, best CV PR-AUC={search.best_score_:.3f}, params={best_params}")
    fitted["Tuned HGB"] = search.best_estimator_
    val_scores["Tuned HGB"] = scores_of(search.best_estimator_, X_val)
    rows.append({"model": "Tuned HGB", **rank_metrics(y_val, val_scores["Tuned HGB"]),
                 "fit seconds": round(time.time() - t0, 1)})
    shortlist = pd.DataFrame(rows).set_index("model").sort_values("PR-AUC", ascending=False)

    # 3. Select ---------------------------------------------------------------------------------------
    candidates = ["Tuned HGB", "Gradient boosting (HGB)", "Random Forest", "Logistic regression", "MLP"]
    ranked = sorted(candidates, key=lambda c: average_precision_score(y_val, val_scores[c]), reverse=True)
    best_name, second = ranked[0], ranked[1]
    rng = np.random.default_rng(seed)
    diffs = np.array([average_precision_score(y_val[i], val_scores[best_name][i])
                      - average_precision_score(y_val[i], val_scores[second][i])
                      for i in stratified_bootstrap_idx(y_val, cfg["evaluation"]["n_bootstrap"], rng)])
    selection = {"selected": best_name, "runner_up": second, "delta_pr_auc_mean": float(diffs.mean()),
                 "delta_pr_auc_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                 "selected_wins_share": float((diffs > 0).mean())}
    log(f"selected {best_name}; vs {second}: dPR-AUC={diffs.mean():+.3f} "
        f"CI [{selection['delta_pr_auc_ci95'][0]:+.3f}, {selection['delta_pr_auc_ci95'][1]:+.3f}]")
    best_model = fitted[best_name]

    # 4. Calibrate ------------------------------------------------------------------------------------
    calibrated = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid").fit(X_val, y_val)
    p_val_raw = scores_of(best_model, X_val)
    p_val = calibrated.predict_proba(X_val)[:, 1]
    calibration = {"brier_raw": brier_score_loss(y_val, p_val_raw), "brier_calibrated": brier_score_loss(y_val, p_val),
                   "mean_p_raw": float(p_val_raw.mean()), "mean_p_calibrated": float(p_val.mean()),
                   "observed_rate": float(y_val.mean())}

    # 5. Decision policy (validation) -------------------------------------------------------------------
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.99, 400), np.quantile(p_val, np.linspace(0.95, 1, 200))]))
    grid_costs = np.array([decision_cost(y_val, p_val >= t, amt_val, c_fp) for t in grid])
    grid_f2 = np.array([fbeta_score(y_val, p_val >= t, beta=2, zero_division=0) for t in grid])
    thresholds = {"cost": float(grid[grid_costs.argmin()]), "f2": float(grid[grid_f2.argmax()])}
    val_policy = policy_report(y_val, p_val, amt_val, c_fp, thresholds)
    chosen = val_policy.drop("No model (approve all)")["cost €"].astype(float).idxmin()
    log(f"policy chosen on validation: {chosen} (thresholds {thresholds})")

    # 6. Test, once -------------------------------------------------------------------------------------
    p_test = calibrated.predict_proba(X_test)[:, 1]
    f_test = apply_policy(chosen, p_test, amt_test, c_fp, thresholds)
    test_policy = policy_report(y_test, p_test, amt_test, c_fp, thresholds)
    test_rank = rank_metrics(y_test, p_test)
    point = {"PR-AUC": test_rank["PR-AUC"], "ROC-AUC": test_rank["ROC-AUC"],
             "saving %": test_policy.loc[chosen, "saving vs no model %"],
             "recall": recall_score(y_test, f_test), "precision": precision_score(y_test, f_test, zero_division=0)}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point}
    for i in stratified_bootstrap_idx(y_test, cfg["evaluation"]["n_bootstrap"], rng):
        yy, pp, aa = y_test[i], p_test[i], amt_test[i]
        f = apply_policy(chosen, pp, aa, c_fp, thresholds)
        boot["PR-AUC"].append(average_precision_score(yy, pp))
        boot["ROC-AUC"].append(roc_auc_score(yy, pp))
        boot["saving %"].append(100 * (1 - decision_cost(yy, f, aa, c_fp) / decision_cost(yy, np.zeros_like(yy), aa, c_fp)))
        boot["recall"].append(recall_score(yy, f)); boot["precision"].append(precision_score(yy, f, zero_division=0))
    test_ci = {k: {"estimate": float(point[k]), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
               for k, v in boot.items()}
    for k, v in test_ci.items():
        log(f"TEST {k:<10} {v['estimate']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")

    frauds = pd.DataFrame({"Amount": amt_test[y_test == 1], "caught": f_test[y_test == 1]})
    frauds["band"] = pd.cut(frauds.Amount, [-0.01, 1, 10, 100, 500, np.inf],
                            labels=["€0-1", "€1-10", "€10-100", "€100-500", "€500+"])
    errors = frauds.groupby("band", observed=False).agg(caught=("caught", "sum"), total=("caught", "size"))
    errors["missed €"] = frauds[~frauds.caught].groupby("band", observed=False)["Amount"].sum()

    # Permutation importance (validation sample)
    sub = pd.concat([X_val.sample(n=min(40000, len(X_val)), random_state=seed), X_val[y_val == 1]]).drop_duplicates()
    pi = permutation_importance(best_model, sub, val.loc[sub.index, TARGET].values, scoring="average_precision",
                                n_repeats=5, random_state=seed, n_jobs=-1)
    importance = pd.Series(pi.importances_mean, index=sub.columns).sort_values(ascending=False)

    # 7. Package -----------------------------------------------------------------------------------------
    X_train_feat = FraudFeatures().transform(X_train)
    metadata = {
        "model_name": best_name, "policy": chosen, "c_fp": c_fp,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "random_state": seed, "train_rows": int(len(train)),
        "train_time_window_s": [float(train.Time.min()), float(train.Time.max())],
        "model_params": {k: v for k, v in best_model.steps[-1][1].get_params().items()
                         if isinstance(v, (int, float, str, bool, type(None)))},
        "test_metrics": {k: round(v["estimate"], 4) for k, v in test_ci.items()},
    }
    bundle = {"model": calibrated, "policy": chosen, "c_fp": c_fp, "thresholds": thresholds,
              "input_columns": RAW_COLS, "reference_bins": {c: make_reference(X_train_feat[c]) for c in OUT_COLS},
              "metadata": metadata}
    joblib.dump(bundle, art / "fraud_model.joblib", compress=3)
    (art / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    metrics = {
        "shortlist_validation": shortlist.round(4).reset_index().to_dict(orient="records"),
        "tuning": {"best_cv_pr_auc": float(search.best_score_), "best_params": best_params},
        "selection": selection, "calibration": calibration, "thresholds": thresholds,
        "validation_policies": val_policy.astype(float).round(4).reset_index(names="policy").to_dict(orient="records"),
        "chosen_policy": chosen,
        "test": test_ci,
        "test_policies": test_policy.astype(float).round(4).reset_index(names="policy").to_dict(orient="records"),
        "test_confusion_matrix": confusion_matrix(y_test, f_test).tolist(),
        "test_missed_frauds_by_amount": errors.reset_index().astype({"band": str}).to_dict(orient="records"),
        "permutation_importance_top10": importance.head(10).round(4).to_dict(),
    }
    (art / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
    pd.concat([X_test.head(3), X_test[y_test == 1].head(2)]).to_csv(art / "sample_request.csv", index=False)
    write_model_card(art, metadata, test_ci, chosen, c_fp, len(train), int(y_train.sum()), int(y_test.sum()))

    make_figures(figs, y_val, val_scores, grid, grid_costs, thresholds, val_policy, y_test, p_test, f_test,
                 point, chosen, importance, shortlist)
    log(f"artifacts written to {art}/, figures to {figs}/ (total {time.time() - t_start:.0f}s)")


def _json_safe(obj):
    """Replace NaN/inf with None and numpy scalars with Python numbers, so metrics.json is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def write_model_card(art, m, test_ci, chosen, c_fp, n_train, train_frauds, test_frauds):
    rows = "\n".join(f"| {k} | {v['estimate']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |" for k, v in test_ci.items())
    (art / "model_card.md").write_text(f"""# Model card: card-fraud decision function

**Model:** {m['model_name']} with Platt calibration | **Decision policy:** {chosen} (false-alarm cost EUR {c_fp:g})
**Trained on:** oldest 60% of the de-duplicated transactions ({n_train:,} rows, {train_frauds} frauds)
**Created:** {m['created']} | scikit-learn {m['sklearn']}

## Intended use
Real-time scoring of card transactions to route suspicious ones to manual review. Not for automatic account closure.

## Test performance (newest 20% of transactions, {test_frauds} frauds; 95% bootstrap CI)
| Metric | Estimate | 95% CI |
|---|---|---|
{rows}

## Limitations
* Two days of data from one issuer (September 2013); fraud patterns change, so retrain and re-validate before real use.
* Features V1-V28 are anonymised PCA components, so feature-level explanations have no business meaning.
* Costs are assumptions; re-derive the threshold if the false-alarm cost changes.
* Labels (chargebacks) arrive with a delay, so live precision and recall are only known weeks later.

## Monitoring
PSI on all inputs against a reference window from the same hours of day, plus mean score and alert rate.
Alert at PSI > 0.25, investigate at > 0.10 (`python -m fraud.monitoring <batch>`).
""")


def make_figures(figs, y_val, val_scores, grid, grid_costs, thresholds, val_policy, y_test, p_test, f_test,
                 point, chosen, importance, shortlist):
    # Shortlist PR curves
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, s in val_scores.items():
        if name.startswith("Dummy"):
            continue
        p, r, _ = precision_recall_curve(y_val, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_val, s):.3f})")
    ax.axhline(y_val.mean(), ls="--", c="grey", label=f"No skill ({y_val.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves (validation)")
    ax.legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(figs / "shortlist_pr_curves.png"); plt.close(fig)

    # Cost vs threshold
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, grid_costs, color="#4C72B0", label="single global threshold")
    ax.axvline(thresholds["cost"], ls="--", c="#4C72B0", alpha=.6)
    ax.axhline(val_policy.loc["Amount-aware Bayes rule", "cost €"], c="#C44E52", ls="-.", label="amount-aware rule")
    ax.axhline(val_policy.loc["No model (approve all)", "cost €"], c="grey", ls=":", label="no model")
    ax.set_xscale("log"); ax.set_xlabel("Threshold on calibrated probability (log)"); ax.set_ylabel("Validation cost (€)")
    ax.set_title("Expected cost vs decision threshold"); ax.legend(); fig.tight_layout()
    fig.savefig(figs / "cost_vs_threshold.png"); plt.close(fig)

    # Test: confusion matrix, PR, reliability
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cm = confusion_matrix(y_test, f_test)
    axes[0].imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm())
    for (i, j), v in np.ndenumerate(cm):
        axes[0].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > 1000 else "black", fontsize=12)
    axes[0].set_xticks([0, 1], ["Approve", "Flag"]); axes[0].set_yticks([0, 1], ["Genuine", "Fraud"])
    axes[0].set_title(f"Test confusion matrix\n({chosen})")
    p, r, _ = precision_recall_curve(y_test, p_test)
    axes[1].plot(r, p, color="#55A868", label=f"AP = {point['PR-AUC']:.3f}")
    axes[1].scatter(point["recall"], point["precision"], c="#C44E52", s=60, zorder=5, label="operating point")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Test precision-recall"); axes[1].legend()
    edges = [0, 1e-3, 1e-2, 0.1, 0.5, 1.0001]
    rel = pd.DataFrame({"b": np.digitize(p_test, edges) - 1, "p": p_test, "y": y_test}).groupby("b").agg(
        pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    axes[2].plot([1e-4, 1], [1e-4, 1], "k--", lw=1, label="perfect calibration")
    axes[2].plot(rel.pred, rel.obs.clip(lower=1e-4), "o-", color="#8172B2", label="model")
    for _, row in rel.iterrows():
        axes[2].annotate(f"n={int(row.n):,}", (row.pred, max(row.obs, 1e-4)), textcoords="offset points",
                         xytext=(6, -12), fontsize=8)
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].legend(fontsize=8, loc="upper left")
    axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Observed fraud rate")
    axes[2].set_title("Reliability diagram (test)")
    fig.tight_layout(); fig.savefig(figs / "test_evaluation.png"); plt.close(fig)

    # Permutation importance
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.head(12)[::-1].plot.barh(ax=ax, color="#DD8452")
    ax.set_xlabel("Drop in PR-AUC when shuffled"); ax.set_title("Permutation importance (validation)")
    fig.tight_layout(); fig.savefig(figs / "permutation_importance.png"); plt.close(fig)


if __name__ == "__main__":
    main()
