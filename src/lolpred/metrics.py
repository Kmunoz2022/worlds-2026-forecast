import numpy as np
import pandas as pd

EPS = 1e-6


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def accuracy(y, p):
    """Favorite = side with p >= 0.5 (ties go to blue)."""
    return float(np.mean((np.asarray(p) >= 0.5) == (np.asarray(y) == 1)))


def summarize(y, p) -> dict:
    return {"n": len(y), "accuracy": accuracy(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p)}


def calibration_table(y, p) -> pd.DataFrame:
    """Folds each game to the favorite's perspective so bins read 50-100%."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    fav_p = np.where(p >= 0.5, p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    edges = [.5, .55, .6, .65, .7, .8, 1.0001]
    b = pd.cut(fav_p, edges, right=False)
    return (pd.DataFrame({"bin": b, "predicted": fav_p, "actual": fav_won})
            .groupby("bin", observed=True).agg(n=("actual", "size"), predicted=("predicted", "mean"),
                                               actual=("actual", "mean")))
