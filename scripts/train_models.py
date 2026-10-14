"""Train and compare game-level models with monthly walk-forward validation.

Every prediction is out-of-sample: for each month from 2022-07, models are
refit on all earlier games and predict that month. Worlds, MSI, EWC and FST
predictions come from the same process, which mirrors real use (retrain
before each month, never see the game being predicted).

Model choices are made on MSI/EWC/FST 2023-2025 ("validation"); Worlds
2022-2025 is reported as the final test.

Usage: .venv/bin/python scripts/train_models.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.elo import EloParams, run_elo  # noqa: E402
from lolpred.features import FEATURES_BASIC, FEATURES_FULL, build_features  # noqa: E402
from lolpred.metrics import calibration_table, log_loss, summarize  # noqa: E402
from lolpred.splits import worlds_backtests  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_MD = ROOT / "docs" / "results" / "models.md"
TRAIN_START = pd.Timestamp("2022-03-01")  # first two months are Elo/form warm-up
WF_START = pd.Timestamp("2022-07-01")
CALIB_MIN_GAMES = 100
DEFAULT_MODEL = "LR full (intl x5) +cal"  # see docs/DECISIONS.md
SWITCH_MARGIN = 0.005


def lr(C):
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def gbdt():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=100, l2_regularization=1.0, random_state=0)


# name: (factory, features, weight on cross-league games)
CONFIGS = {
    "LR basic": (lambda: lr(1.0), FEATURES_BASIC, 1.0),
    "LR full": (lambda: lr(0.1), FEATURES_FULL, 1.0),
    "LR full (intl x5)": (lambda: lr(0.1), FEATURES_FULL, 5.0),
    "GBDT full": (gbdt, FEATURES_FULL, 1.0),
    "GBDT full (intl x5)": (gbdt, FEATURES_FULL, 5.0),
}


def walk_forward_predict(X, y, dates, cross, factory, feats, w_cross):
    out = np.full(len(X), np.nan)
    months = pd.date_range(WF_START, dates.max() + pd.offsets.MonthBegin(1), freq="MS")
    for a, b in zip(months[:-1], months[1:]):
        test = (dates >= a) & (dates < b)
        if not test.any():
            continue
        train = (dates >= TRAIN_START) & (dates < a)
        model = factory()
        w = np.where(cross[train], w_cross, 1.0)
        model.fit(X.loc[train, feats], y[train], **_weight_kw(model, w))
        out[test] = model.predict_proba(X.loc[test, feats])[:, 1]
    return out


def _weight_kw(model, w):
    if hasattr(model, "steps"):
        return {f"{model.steps[-1][0]}__sample_weight": w}
    return {"sample_weight": w}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_slope(p, y):
    """Temperature scaling: one multiplier on the logit, chosen by log loss."""
    z = _logit(p)
    grid = np.arange(0.5, 1.51, 0.02)
    losses = [log_loss(y, 1 / (1 + np.exp(-a * z))) for a in grid]
    return float(grid[int(np.argmin(losses))])


def calibrate_walk_forward(p, y, dates, calib_mask):
    """Rescale each month's predictions with a slope fit on earlier
    out-of-sample international predictions (calib_mask). Months without
    enough earlier games stay unscaled."""
    out = p.copy()
    slopes = {}
    months = pd.date_range(WF_START, dates.max() + pd.offsets.MonthBegin(1), freq="MS")
    for a, b in zip(months[:-1], months[1:]):
        test = (dates >= a) & (dates < b)
        prior = calib_mask & (dates < a) & ~np.isnan(p)
        if not test.any() or prior.sum() < CALIB_MIN_GAMES:
            continue
        s = fit_slope(p[prior], y[prior])
        slopes[f"{a:%Y-%m}"] = s
        out[test] = 1 / (1 + np.exp(-s * _logit(p[test])))
    return out, slopes


def main():
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values(["date", "gameid"],
                                                                        kind="stable").reset_index(drop=True)
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    elo_params = EloParams(**json.loads((MODELS_DIR / "elo_params.json").read_text())["full"])
    elo_preds = run_elo(m, elo_params)
    X = build_features(m, tg, elo_preds)
    y = m["blue_win"].to_numpy()
    dates = m["date"]
    cross = m["cross_league"].to_numpy()

    preds = {"Elo": elo_preds["p_blue"].to_numpy()}
    for name, (factory, feats, w) in CONFIGS.items():
        print(f"walk-forward: {name}")
        preds[name] = walk_forward_predict(X, y, dates, cross, factory, feats, w)

    intl = m["league"].isin(["MSI", "EWC", "FST", "WLDs"]).to_numpy()
    slopes = {}
    for name in list(preds):
        preds[f"{name} +cal"], slopes[name] = calibrate_walk_forward(preds[name], y, dates, intl)

    # --- evaluation sets (all out-of-sample)
    d23 = (dates >= "2023-01-01").to_numpy()
    sets = {
        "all games 2023+": d23,
        "cross-league 2023+": d23 & cross,
        "VALIDATION: MSI/EWC/FST 2023-25": d23 & m["league"].isin(["MSI", "EWC", "FST"]).to_numpy()
                                           & (dates < "2026-01-01").to_numpy(),
    }
    worlds = np.zeros(len(m), bool)
    for sp in worlds_backtests(m):
        idx = sp.test(m).index
        worlds[idx] = True
        sets[f"TEST: {sp.name}"] = np.isin(np.arange(len(m)), idx)
    sets["TEST: Worlds 2022-25 pooled"] = worlds

    rows = []
    for set_name, mask in sets.items():
        for name, p in preds.items():
            ok = mask & ~np.isnan(p)
            rows.append({"set": set_name, "model": name, **summarize(y[ok], p[ok])})
    res = pd.DataFrame(rows)
    table = res.set_index(["set", "model"])[["n", "accuracy", "log_loss", "brier"]].round(3)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(table.to_string())

    # --- pick the model on validation log loss, report its Worlds numbers.
    # Keep the default (simpler, explainable) model unless a challenger beats
    # it by more than SWITCH_MARGIN: 321 validation games can't separate
    # smaller differences from noise.
    val = res[res["set"].str.startswith("VALIDATION")].set_index("model")["log_loss"]
    challenger = val.idxmin()
    best = challenger if val[challenger] < val[DEFAULT_MODEL] - SWITCH_MARGIN else DEFAULT_MODEL
    print(f"\nvalidation: default {DEFAULT_MODEL} {val[DEFAULT_MODEL]:.4f}, "
          f"best {challenger} {val[challenger]:.4f} -> using {best}")
    cal = calibration_table(y[worlds], preds[best][worlds]).round(3)
    print(f"\n== calibration on Worlds 2022-25, {best}\n{cal.to_string()}")

    # --- fit the chosen model on all data for live use
    base = best.replace(" +cal", "")
    final_meta = {"model": base, "calibrated": best.endswith("+cal"),
                  "calibration_slope": slopes.get(base, {}) and list(slopes[base].values())[-1],
                  "trained_through": f"{dates.max():%Y-%m-%d}"}
    if base != "Elo":
        factory, feats, w = CONFIGS[base]
        train = (dates >= TRAIN_START).to_numpy()
        final = factory()
        final.fit(X.loc[train, feats], y[train], **_weight_kw(final, np.where(cross[train], w, 1.0)))
        joblib.dump({"model": final, "features": feats}, MODELS_DIR / "game_model.joblib")
        final_meta["features"] = feats
        if hasattr(final, "steps"):
            coefs = pd.Series(final.steps[-1][1].coef_[0], index=feats).sort_values(key=abs, ascending=False)
            print(f"\n== standardized coefficients ({base})\n{coefs.round(3).to_string()}")
            final_meta["coefficients"] = coefs.round(4).to_dict()
    (MODELS_DIR / "game_model.json").write_text(json.dumps(final_meta, indent=2))

    out = m[["gameid", "date", "league", "blue_teamid", "red_teamid", "blue_win", "cross_league"]].copy()
    for name, p in preds.items():
        out[f"p_{name}"] = p
    out.to_parquet(PROCESSED_DIR / "model_predictions.parquet", index=False)
    X.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(
        "# Game model comparison\n\nGenerated by `scripts/train_models.py`. All predictions are "
        "out-of-sample (monthly walk-forward). `+cal` = temperature-scaled on earlier international "
        "predictions. Choice made on VALIDATION; TEST rows are Worlds.\n\n"
        f"Chosen model: **{best}**\n\n```\n{table.to_string()}\n```\n\n"
        f"## Calibration on Worlds 2022-25 ({best})\n\n```\n{cal.to_string()}\n```\n\n"
        f"## Final model\n\n```json\n{json.dumps(final_meta, indent=2)}\n```\n")
    print(f"\nwrote {RESULTS_MD}, models/game_model.*, {PROCESSED_DIR}/model_predictions.parquet")


if __name__ == "__main__":
    main()
