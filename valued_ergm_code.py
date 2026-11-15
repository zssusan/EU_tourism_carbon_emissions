#!/usr/bin/env python3
"""Valued ERGM from the file `data` in this folder.

Running this script reads `data` and writes `result` next to it. Both files
are CSV. `data` is the estimation sample; `result` is overwritten on each run.

For each of 2012, 2016, 2019 and 2024 it estimates a directed valued ERGM
on the 702 international edges.

Dependent variable
    Y in {0, 1, 2, 3, 4}, already stored in the CSV.

Reference
    Binomial with 4 trials. The dyad pseudolikelihood is

        P(Y_ij = k) ∝ C(4, k) * exp(theta · z_ij(k))

    where z_ij(k) is the change statistic at value k, holding the rest of
    the observed network fixed. This is maximum pseudolikelihood, the
    estimator that produced the reported sender/receiver table.

Terms, in order
    Sum                         k
    sqrt popularity             sqrt(k) * (sqrt in-weight of j, excluding ij) * 2/(n-2)
    geometric reciprocity       sqrt(k) * sqrt(Y_ji) when Y_ji > 0
    nodeocov ln(LOS)            k * ln(LOS of the origin)      LOS_sender
    nodeicov ln(LOS)            k * ln(LOS of the destination) LOS_receiver
    nodeocov ln(SEAS)           k * ln(SEAS_sender)
    nodeicov ln(SEAS)           k * ln(SEAS_receiver)
    nodeocov ln(EMPL)           k * ln(EMPL_sender)
    nodeicov ln(EMPL)           k * ln(EMPL_receiver)

LOS, SEAS and EMPL in the CSV are the original country values. The model
uses their natural logs.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from numpy.linalg import eigvalsh, inv
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "data"
OUT_PATH = HERE / "result"

EU27 = [
    "Belgium", "Bulgaria", "Czechia", "Denmark", "Germany", "Estonia",
    "Ireland", "Greece", "Spain", "France", "Croatia", "Italy", "Cyprus",
    "Latvia", "Lithuania", "Luxembourg", "Hungary", "Malta", "Netherlands",
    "Austria", "Poland", "Portugal", "Romania", "Slovenia", "Slovakia",
    "Finland", "Sweden",
]
YEARS = (2012, 2016, 2019, 2024)
N = 27
N_DYAD = N * (N - 1)
IDX = {name: i for i, name in enumerate(EU27)}
LOGC = np.array([
    math.lgamma(5) - math.lgamma(k + 1) - math.lgamma(5 - k) for k in range(5)
])
LABELS = [
    "Sum",
    "popularity",
    "reciprocity",
    "nodeocov ln(LOS)",
    "nodeicov ln(LOS)",
    "nodeocov ln(SEAS)",
    "nodeicov ln(SEAS)",
    "nodeocov ln(EMPL)",
    "nodeicov ln(EMPL)",
]


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def load_year(rows, year):
    """Y and the logged node covariates for one year, taken from the CSV columns."""
    disc = np.full((N, N), -1, dtype=int)
    los = np.full(N, np.nan)
    seas = np.full(N, np.nan)
    empl = np.full(N, np.nan)
    for row in rows:
        if int(row["year"]) != year:
            continue
        i = IDX[row["FROM_country"]]
        j = IDX[row["TO_country"]]
        disc[i, j] = int(row["Y"])
        los_i = math.log(float(row["LOS_sender"]))
        seas_i = math.log(float(row["SEAS_sender"]))
        empl_i = math.log(float(row["EMPL_sender"]))
        los_j = math.log(float(row["LOS_receiver"]))
        seas_j = math.log(float(row["SEAS_receiver"]))
        empl_j = math.log(float(row["EMPL_receiver"]))
        for arr, idx, val in (
            (los, i, los_i), (seas, i, seas_i), (empl, i, empl_i),
            (los, j, los_j), (seas, j, seas_j), (empl, j, empl_j),
        ):
            if np.isnan(arr[idx]):
                arr[idx] = val
            elif not np.isclose(arr[idx], val):
                raise SystemExit(f"inconsistent covariate at {row['FROM_country']}->{row['TO_country']} {year}")
    if np.any(disc[~np.eye(N, dtype=bool)] < 0):
        raise SystemExit(f"missing edges in {year}")
    if np.any(np.isnan(los)) or np.any(np.isnan(seas)) or np.any(np.isnan(empl)):
        raise SystemExit(f"missing covariates in {year}")
    np.fill_diagonal(disc, 0)
    return disc, los, seas, empl


def design(disc, ln_los, ln_seas, ln_empl):
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    k = np.arange(5, dtype=float)
    sqrt_k = np.sqrt(k)
    scale = 2.0 / (N - 2)
    sqrt_y = np.sqrt(disc.astype(float))
    sqrt_in = sqrt_y.sum(axis=0)
    z = np.zeros((N_DYAD, 5, len(LABELS)))
    yobs = np.empty(N_DYAD, dtype=int)
    for d, (i, j) in enumerate(pairs):
        yobs[d] = int(disc[i, j])
        other = sqrt_in[j] - sqrt_y[i, j]
        rev = float(disc[j, i])
        z[d, :, 0] = k
        z[d, :, 1] = sqrt_k * scale * other
        z[d, :, 2] = sqrt_k * np.sqrt(rev) if rev > 0 else 0.0
        z[d, :, 3] = k * ln_los[i]
        z[d, :, 4] = k * ln_los[j]
        z[d, :, 5] = k * ln_seas[i]
        z[d, :, 6] = k * ln_seas[j]
        z[d, :, 7] = k * ln_empl[i]
        z[d, :, 8] = k * ln_empl[j]
    return z, yobs


def fit(z, yobs):
    def nll(th):
        eta = LOGC[None, :] + z @ th
        ll = eta[np.arange(N_DYAD), yobs] - logsumexp(eta, axis=1)
        return -float(ll.sum())

    def grad(th):
        eta = LOGC[None, :] + z @ th
        eta = eta - eta.max(axis=1, keepdims=True)
        p = np.exp(eta)
        p /= p.sum(axis=1, keepdims=True)
        obs = z[np.arange(N_DYAD), yobs]
        mean = (p[:, :, None] * z).sum(axis=1)
        return -(obs - mean).sum(axis=0)

    def hess(th):
        eta = LOGC[None, :] + z @ th
        eta = eta - eta.max(axis=1, keepdims=True)
        p = np.exp(eta)
        p /= p.sum(axis=1, keepdims=True)
        mean = (p[:, :, None] * z).sum(axis=1)
        h = np.zeros((len(th), len(th)))
        for d in range(N_DYAD):
            zd = z[d]
            mu = mean[d]
            w = p[d]
            h += (zd * w[:, None]).T @ zd - np.outer(mu, mu)
        return h

    res = minimize(
        nll, np.zeros(z.shape[2]), method="BFGS", jac=grad,
        options={"maxiter": 400, "gtol": 1e-8},
    )
    grad_norm = float(np.linalg.norm(res.jac))
    if grad_norm > 1e-5:
        raise SystemExit(f"pseudolikelihood did not converge: {res.message}; grad {grad_norm:.3e}")
    h = hess(res.x)
    se = np.sqrt(np.clip(np.diag(inv(h)), 0, np.inf))
    rows = []
    for i, name in enumerate(LABELS):
        zstat = float(res.x[i] / se[i])
        p = float(2 * norm.sf(abs(zstat)))
        rows.append({
            "term": name,
            "coef": float(res.x[i]),
            "se": float(se[i]),
            "z": zstat,
            "p": p,
            "stars": stars(p),
        })
    return rows, float(-res.fun)


def main():
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"read {CSV_PATH}")
    print(f"rows {len(rows)}")

    fitted = {}
    out_rows = []
    for year in YEARS:
        disc, ln_los, ln_seas, ln_empl = load_year(rows, year)
        est, ll = fit(*design(disc, ln_los, ln_seas, ln_empl))
        fitted[year] = est
        print(year, "log pseudolikelihood", round(ll, 2))
        for row in est:
            out_rows.append({"year": year, **row})

    print()
    print(f"{'term':<22}" + "".join(f"{year:>22}" for year in YEARS))
    for name in LABELS:
        line = f"{name:<22}"
        for year in YEARS:
            cell = next(r for r in fitted[year] if r["term"] == name)
            line += f"{cell['coef']:8.3f}{cell['stars']:<3}({cell['se']:.3f})".rjust(22)
        print(line)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "term", "coef", "se", "z", "p", "stars"])
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
