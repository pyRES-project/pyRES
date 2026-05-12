"""Error metrics for comparing a model series against a reference series."""
import numpy as np
import pandas as pd


def _align(pred, ref):
    pred = pd.Series(pred).astype(float)
    ref = pd.Series(ref).astype(float)
    df = pd.concat([pred.rename('pred'), ref.rename('ref')], axis=1).dropna()
    return df['pred'].values, df['ref'].values


def rmse(pred, ref):
    p, r = _align(pred, ref)
    return float(np.sqrt(np.mean((p - r) ** 2)))


def mae(pred, ref):
    p, r = _align(pred, ref)
    return float(np.mean(np.abs(p - r)))


def mape(pred, ref, eps=1e-6):
    p, r = _align(pred, ref)
    mask = np.abs(r) > eps
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((p[mask] - r[mask]) / r[mask])) * 100)


def nmbe(pred, ref):
    p, r = _align(pred, ref)
    denom = np.mean(r)
    if denom == 0:
        return float('nan')
    return float(np.mean(p - r) / denom * 100)


def r2(pred, ref):
    p, r = _align(pred, ref)
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1 - ss_res / ss_tot)


def summary(pred, ref):
    return pd.DataFrame({
        'metric': ['RMSE', 'MAE', 'MAPE [%]', 'NMBE [%]', 'R2'],
        'value': [rmse(pred, ref), mae(pred, ref), mape(pred, ref),
                  nmbe(pred, ref), r2(pred, ref)],
    })
