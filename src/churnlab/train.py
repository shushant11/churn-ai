"""End-to-end training:  python -m churnlab.train
1 load+validate  2 split 60/20/20  3 fit every candidate on TRAIN  4 pick the best by VALIDATION PR-AUC
5 calibrate on VALIDATION  6 choose the threshold by expected profit on VALIDATION  7 touch TEST once  8 save everything."""
from __future__ import annotations
import argparse, json, time, warnings
import numpy as np
from . import config, plots
from .business import best_threshold, decile_table, profit_curve, top_k_capture
from .data import load, split, validate, xy
from .evaluate import PlattCalibrator, at_threshold, bootstrap_ci, f1_threshold, ranking
from .explain import global_importance
from .fairness import audit
from .models import Blend, candidates, tuned_hist_gb, tuned_networks, with_gender
from .predict import ModelBundle
from .survival import by_group, logrank
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


def _mlflow():
    try:
        import mlflow
        mlflow.set_experiment("churn-ai")
        return mlflow
    except ImportError:
        return None


def main(n_trials: int = 25, fast: bool = False, nn_trials: int = 10, keep_gender: bool = False):
    t0 = time.time()
    config.DROP_GENDER = not keep_gender                 # gender-blind unless asked otherwise (see MODEL_CARD)
    df = load()
    tr, va, te = split(df)
    Xtr, ytr = xy(tr); Xva, yva = xy(va); Xte, yte = xy(te)
    print(f"rows train/val/test = {len(tr)}/{len(va)}/{len(te)}   churn rate = {df[config.TARGET].mean():.3f}")

    # ---- 3-4  model zoo, selection on validation ------------------------------------------------
    models = candidates()
    pre_fitted, nn_tuning = set(), {}
    if not fast:
        tuned, tune_info = tuned_hist_gb(Xtr, ytr, n_trials=n_trials)
        models["hist_gb_tuned"] = tuned; pre_fitted.add("hist_gb_tuned")
        if nn_trials > 0:                               # tuned neural nets / transformer (search on TRAIN only)
            for name, (est, info) in tuned_networks(Xtr, ytr, nn_trials).items():
                models[name] = est; pre_fitted.add(name); nn_tuning[name] = info
                print(f"  tuned {name}: CV PR-AUC {info['cv_pr_auc']:.4f}  {info['best_params']}")
    rows, fitted = [], {}

    def score(name, m, secs):
        p = m.predict_proba(Xva)[:, 1]
        rows.append({"model": name, "val_pr_auc": round(float(average_precision_score(yva, p)), 4),
                     "val_roc_auc": round(float(roc_auc_score(yva, p)), 4), "fit_seconds": round(secs, 1)})
        fitted[name] = m
        print(f"  {name:20s} val PR-AUC {rows[-1]['val_pr_auc']:.4f}  ROC-AUC {rows[-1]['val_roc_auc']:.4f}")

    for name, m in models.items():
        t1 = time.time()
        if name not in pre_fitted:
            m.fit(Xtr, ytr)
        score(name, m, time.time() - t1)

    # blends: members are the best single models by VALIDATION PR-AUC (equal weights, nothing fitted on the test set)
    singles = sorted(rows, key=lambda r: r['val_pr_auc'], reverse=True)
    trees = [r['model'] for r in singles if r['model'] in {'random_forest', 'hist_gb', 'hist_gb_tuned', 'xgboost', 'lightgbm'}]
    nets = [r['model'] for r in singles if r['model'] in {'mlp_sklearn', 'ann_pytorch', 'ann_pytorch_tuned', 'ann_keras', 'ann_keras_tuned', 'ft_transformer', 'ft_transformer_tuned'}]
    plans = {'blend_top3': [r['model'] for r in singles[:3]], 'blend_trees_nets': trees[:2] + nets[:2]}
    seen = set()
    for bname, members in plans.items():
        key = frozenset(members)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        score(bname, Blend.from_fitted(members, [fitted[k] for k in members]), 0.0)
        print(f"      {bname} = {' + '.join(members)}")
    rows.sort(key=lambda r: r["val_pr_auc"], reverse=True)
    best = rows[0]["model"]; model = fitted[best]
    print(f"selected: {best}")

    # ---- 5-6  calibrate + threshold on validation --------------------------------------------
    raw_va = model.predict_proba(Xva)[:, 1]
    cal = PlattCalibrator().fit(raw_va, yva)
    p_va = cal.transform(raw_va)
    t_profit, t_f1 = best_threshold(yva, p_va), f1_threshold(yva, p_va)

    # ---- 7  the ONE look at the test set ------------------------------------------------------
    raw_te = model.predict_proba(Xte)[:, 1]
    p_te = cal.transform(raw_te)
    rk = ranking(yte, p_te)
    ci = {"pr_auc_95ci": bootstrap_ci(yte, p_te, average_precision_score), "roc_auc_95ci": bootstrap_ci(yte, p_te, roc_auc_score)}
    thresholds = {"default_0.5": at_threshold(yte, p_te, 0.5), "f1_opt": at_threshold(yte, p_te, t_f1), "profit_opt": at_threshold(yte, p_te, t_profit)}
    # "contact everyone" baseline: what a bank would do without a model
    everyone = at_threshold(yte, np.ones_like(p_te), 0.5)
    dec = decile_table(yte, p_te)
    baseline_acc = 1 - yte.mean()

    # ---- extras: fairness, survival, importance -------------------------------------------------
    fair = audit(te.reset_index(drop=True), yte, p_te, t_profit)
    # gender comparison: refit the SAME model with the opposite gender choice and compare accuracy and the audit
    alt = with_gender(model, use_gender=config.DROP_GENDER).fit(Xtr, ytr)
    pa_va = alt.predict_proba(Xva)[:, 1]; cal_a = PlattCalibrator().fit(pa_va, yva)
    pa_te = cal_a.transform(alt.predict_proba(Xte)[:, 1])
    fair_alt = audit(te.reset_index(drop=True), yte, pa_te, best_threshold(yva, cal_a.transform(pa_va)), attrs=("Gender",))
    fair["gender_comparison"] = {"selected_model_uses_gender": not config.DROP_GENDER,
                                 "selected": {"test_pr_auc": rk["pr_auc"], "gender_disparity": fair["Gender"]["disparity"]},
                                 "alternative": {"uses_gender": config.DROP_GENDER, "test_pr_auc": round(float(average_precision_score(yte, pa_te)), 4),
                                                 "gender_disparity": fair_alt["Gender"]["disparity"]}}
    imp = global_importance(model, Xte, yte, n_repeats=5 if fast else 10)
    km_geo, km_act = by_group(df, "Geography"), by_group(df.assign(Active=df.IsActiveMember.map({1: "active", 0: "inactive"})), "Active")
    lr_geo = logrank(df, "Geography")

    metrics = {
        "dataset": {"gender_blind": config.DROP_GENDER, "rows": len(df), "churn_rate": round(float(df[config.TARGET].mean()), 4), "split": "stratified 60/20/20, seed 42"},
        "model_selection": rows, "selected_model": best,
        "tuning": tune_info if not fast else None, "tuning_neural": nn_tuning or None,
        "blend_members": {k: list(v.names) for k, v in fitted.items() if isinstance(v, Blend)},
        "test": {**{k: round(v, 4) for k, v in rk.items()}, **{k: [round(a, 4) for a in v] for k, v in ci.items()},
                 "raw_brier": round(float(np.mean((raw_te - yte) ** 2)), 4),
                 "accuracy_at_0.5": round(float(((p_te >= 0.5) == yte).mean()), 4), "majority_class_accuracy": round(float(baseline_acc), 4)},
        "thresholds": thresholds, "contact_everyone_baseline": everyone,
        "top20pct_capture": round(top_k_capture(yte, p_te, 0.2), 4),
        "deciles": dec.round(4).to_dict(orient="records"),
        "business_assumptions": vars(config.BUSINESS),
        "fairness": fair, "importance": imp.round(4).to_dict(orient="records"),
        "survival_logrank_geography": lr_geo,
        "seconds": round(time.time() - t0, 1),
    }
    (config.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    plots.model_compare(rows)
    plots.pr_roc({best: p_te, **({"logreg (baseline)": fitted["logreg"].predict_proba(Xte)[:, 1]} if best != "logreg" else {})}, yte, yte.mean())
    plots.calibration(yte, raw_te, p_te)
    plots.gain(dec)
    plots.profit(profit_curve(yte, p_te), t_profit)
    plots.importance(imp)
    plots.km(km_geo, "Survival by Geography"); plots.km(km_act, "Survival by Activity")

    bundle = ModelBundle(pipeline=model, calibrator=cal, threshold=t_profit, model_name=best,
                         background=Xtr.sample(500, random_state=config.SEED).reset_index(drop=True),
                         metadata={"uses_gender": not config.DROP_GENDER, "test_pr_auc": rk["pr_auc"], "test_roc_auc": rk["roc_auc"], "trained_rows": len(tr),
                                   "sklearn": __import__("sklearn").__version__, "churnlab": __import__("churnlab").__version__})
    bundle.save()
    (config.ARTIFACTS / "reference.csv").write_text(Xtr.to_csv(index=False), encoding="utf-8")       # drift reference
    te.to_csv(config.ARTIFACTS / "test_holdout.csv", index=False)

    ml = _mlflow()
    if ml:
        with ml.start_run(run_name=best):
            ml.log_params({"model": best, "threshold": t_profit, **(tune_info["best_params"] if not fast else {})})
            ml.log_metrics({"test_pr_auc": rk["pr_auc"], "test_roc_auc": rk["roc_auc"], "test_brier": rk["brier"], "test_ece": rk["ece"],
                            "profit_opt_profit": thresholds["profit_opt"]["profit"]})
            ml.log_artifacts(str(config.FIGURES)); ml.log_artifact(str(config.REPORTS / "metrics.json"))

    print(json.dumps({"test": metrics["test"], "profit_opt": thresholds["profit_opt"]}, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--nn-trials", type=int, default=10, help="random-search trials per neural model (0 = skip)")
    ap.add_argument("--fast", action="store_true", help="skip tuning (for CI)")
    ap.add_argument("--keep-gender", action="store_true", help="let the model see Gender (default: gender-blind)")
    a = ap.parse_args()
    main(a.trials, a.fast, a.nn_trials, a.keep_gender)
