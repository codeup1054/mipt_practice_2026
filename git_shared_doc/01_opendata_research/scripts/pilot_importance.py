"""Пилот H5-H8 на NHANES 2021-2023: важность блоков привычек для ИМТ.

Что делает:
  * собирает когорту взрослых 20+ с достоверным 24h recall и измеренным ИМТ;
  * строит 5 блоков признаков: демография, питание, сон, активность,
    психологический дистресс (PHQ-9 — прокси стресса, прямого вопроса нет);
  * учит RandomForest на ИМТ, считает важность блока как падение R2 при
    совместной перестановке всех колонок блока (10 повторов);
  * bootstrap по тестовым строкам (200 повторов) -> 95% CI важности блока;
  * повторяет на 3 seed'ах (H7) и считает OLS-коэффициенты с bootstrap-CI
    как «традиционный способ» (H6);
  * проверяет H8: доля блока, которую объясняют 1-2 ведущих признака.

Запуск:
    PYTHONIOENCODING=utf-8 python scripts/pilot_importance.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
NH = ROOT / "processed" / "nhanes"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 7, 2024]
N_PERM = 10
N_BOOT = 200
BLOCKS = ["demo", "diet", "sleep", "activity", "distress"]


def load(mod: str, cols: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(NH / f"{mod}_L.parquet", columns=cols).to_pandas()


def hhmm_to_hours(s: pd.Series) -> pd.Series:
    """'21:30' -> 21.5; нечисловые коды (77777/99999) -> NaN."""
    t = pd.to_datetime(s.astype("string"), format="%H:%M", errors="coerce")
    return t.dt.hour + t.dt.minute / 60


def build_cohort() -> pd.DataFrame:
    demo = load("DEMO")
    bmx = load("BMX")
    bp = load("BPXO")
    slq = load("SLQ")
    paq = load("PAQ")
    dr = load("DR1TOT")
    dpq = load("DPQ")

    d = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2",
              "INDFMPIR", "WTMEC2YR"]].copy()
    d = d.merge(bmx[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN", how="left")
    d = d.merge(bp[["SEQN", "BPXOSY1", "BPXODI1", "BPXOPLS1"]], on="SEQN", how="left")
    d = d.merge(slq[["SEQN", "SLD012", "SLD013", "SLQ300", "SLQ310"]], on="SEQN",
                how="left")
    d = d.merge(paq[["SEQN", "PAD790Q", "PAD790U", "PAD800", "PAD810Q",
                     "PAD810U", "PAD820", "PAD680"]], on="SEQN", how="left")
    d = d.merge(dr[["SEQN", "DR1DRSTZ", "DR1TKCAL", "DR1TPROT", "DR1TCARB",
                    "DR1TTFAT", "DR1TFIBE", "DR1TSODI", "DR1TPOTA", "DR1TVC",
                    "DR1_320Z"]], on="SEQN", how="left")
    d = d.merge(dpq[["SEQN"] + [f"DPQ0{i}0" for i in range(1, 10)]], on="SEQN",
                how="left")

    # --- когорта
    d = d[(d.RIDAGEYR >= 20)
          & d.BMXBMI.between(12, 80)
          & (d.DR1DRSTZ == 1)
          & d.DR1TKCAL.between(400, 8000)
          & d.BPXOSY1.between(50, 260)]
    d = d.copy()

    # --- демография
    d["female"] = (d.RIAGENDR == 2).astype(float)
    for code in (1, 2, 4, 6, 7):
        d[f"race_{code}"] = (d.RIDRETH3 == code).astype(float)
    d["educ"] = d.DMDEDUC2.where(d.DMDEDUC2 <= 5)      # 7/9 -> NaN
    d["pir"] = d.INDFMPIR

    # --- питание: доли энергии, плотность на 1000 ккал
    kcal = d.DR1TKCAL.replace(0, np.nan)
    d["log_kcal"] = np.log1p(kcal)
    d["prot_per1000"] = d.DR1TPROT / kcal * 1000
    d["carb_pct"] = d.DR1TCARB * 4 / kcal * 100
    d["fat_pct"] = d.DR1TTFAT * 9 / kcal * 100
    d["fiber_per1000"] = d.DR1TFIBE / kcal * 1000
    d["sodium_per1000"] = d.DR1TSODI / kcal * 1000
    d["potassium_per1000"] = d.DR1TPOTA / kcal * 1000
    d["water_per1000"] = d.DR1_320Z / kcal * 1000

    # --- сон
    d["sleep_weekday"] = d.SLD012
    d["sleep_weekend"] = d.SLD013
    d["sleep_debt"] = d.SLD013 - d.SLD012
    bed = hhmm_to_hours(d.SLQ300)
    wake = hhmm_to_hours(d.SLQ310)
    dur = (wake - bed) % 24
    d["sleep_midpoint"] = (bed + dur / 2) % 24

    # --- активность (минуты в неделю)
    def weekly(q: pd.Series, unit: pd.Series) -> pd.Series:
        q = q.where(q < 9999)
        f = unit.map({"D": 7.0, "W": 1.0, "M": 1 / 4.345, "Y": 1 / 52.14})
        f = f.fillna(0.0)          # пустая единица = не делал
        return (q * f).clip(lower=0, upper=7 * 4)

    mod_days = weekly(d.PAD790Q, d.PAD790U)
    vig_days = weekly(d.PAD810Q, d.PAD810U)
    d["mod_min_wk"] = (mod_days * d.PAD800.where(d.PAD800 < 9999)).clip(0, 3000)
    d["vig_min_wk"] = (vig_days * d.PAD820.where(d.PAD820 < 9999)).clip(0, 3000)
    d["mvpa_min_wk"] = d.mod_min_wk.fillna(0) + d.vig_min_wk.fillna(0)
    d["sedentary_min_day"] = d.PAD680.where(d.PAD680 < 9999)

    # --- дистресс (PHQ-9): 0-3, отказ 7/9 -> NaN
    phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    d["phq9"] = d[phq_cols].where(d[phq_cols] <= 3).sum(axis=1, min_count=7)

    return d


FEATURES = {
    "demo": ["RIDAGEYR", "female", "pir", "educ", "race_1", "race_2",
             "race_4", "race_6", "race_7"],
    # белок — плотность на 1000 ккал, чтобы не было идеальной суммы долей
    "diet": ["log_kcal", "prot_per1000", "carb_pct", "fat_pct", "fiber_per1000",
             "sodium_per1000", "potassium_per1000", "water_per1000"],
    "sleep": ["sleep_weekday", "sleep_weekend", "sleep_debt", "sleep_midpoint"],
    # mvpa_min_wk = mod + vig: производная, в модель не берём (коллинеарность)
    "activity": ["mod_min_wk", "vig_min_wk", "sedentary_min_day"],
    "distress": ["phq9"],
}
LABELS = {"demo": "демография", "diet": "питание", "sleep": "сон",
          "activity": "активность", "distress": "дистресс (PHQ-9)"}


def block_bootstrap(pred_base: np.ndarray, pred_block: dict[str, np.ndarray],
                    y: np.ndarray, seed: int) -> pd.DataFrame:
    """Важность блока (ΔR²) + bootstrap-CI по тестовым строкам."""
    rng = np.random.default_rng(seed)
    n = len(y)
    base_err = (y - pred_base) ** 2
    perm_err = {b: [((y - p) ** 2) for p in preds] for b, preds in pred_block.items()}

    rows = []
    boot = {b: [] for b in BLOCKS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        sst = ((yy - yy.mean()) ** 2).sum()
        r2_base = 1 - base_err[idx].sum() / sst
        for b in BLOCKS:
            r2p = np.mean([1 - e[idx].sum() / sst for e in perm_err[b]])
            boot[b].append(r2_base - r2p)

    for b in BLOCKS:
        v = np.array(boot[b])
        rows.append({
            "block": b,
            "block_ru": LABELS[b],
            "delta_r2": round(float(v.mean()), 5),
            "ci_lo": round(float(np.percentile(v, 2.5)), 5),
            "ci_hi": round(float(np.percentile(v, 97.5)), 5),
        })
    df = pd.DataFrame(rows).sort_values("delta_r2", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df


def run_seed(df: pd.DataFrame, seed: int, target: str = "BMXBMI") -> dict:
    cols = [c for f in FEATURES.values() for c in f]
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = df[target].astype(float).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)

    rng = np.random.default_rng(seed)
    pred_block = {}
    for b in BLOCKS:
        preds = []
        for _ in range(N_PERM):
            Xp = Xte.copy()
            idx = rng.permutation(len(Xp))
            b_cols = FEATURES[b]
            Xp[b_cols] = Xte[b_cols].values[idx]
            preds.append(rf.predict(Xp))
        pred_block[b] = preds

    blocks = block_bootstrap(rf.predict(Xte), pred_block, yte, seed)
    blocks["seed"] = seed
    blocks["target"] = target
    blocks["r2_test"] = round(r2_score(yte, rf.predict(Xte)), 4)
    blocks["n_train"] = len(Xtr)
    blocks["n_test"] = len(Xte)
    return {"blocks": blocks, "model": rf, "Xte": Xte, "yte": yte,
            "features": cols}


def feature_importance(model, Xte, yte, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = r2_score(yte, model.predict(Xte))
    rows = []
    for b, cols in FEATURES.items():
        for c in cols:
            vals = []
            for _ in range(5):
                Xp = Xte.copy()
                Xp[c] = Xte[c].values[rng.permutation(len(Xp))]
                vals.append(base - r2_score(yte, model.predict(Xp)))
            rows.append({"block": b, "feature": c,
                         "delta_r2": round(float(np.mean(vals)), 6)})
    df = pd.DataFrame(rows)
    df["rank_in_block"] = df.groupby("block")["delta_r2"].rank(
        ascending=False, method="first").astype(int)
    return df


def pdp_direction(model, Xte: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Направление связи по ML: знак наклона partial dependence (q10 -> q90)."""
    rows = []
    for b, cols in FEATURES.items():
        for c in cols:
            grid = np.linspace(Xte[c].quantile(.1), Xte[c].quantile(.9), 15)
            Xp = pd.DataFrame({k: np.full(len(grid), Xte[k].median())
                               for k in Xte.columns})
            Xp[c] = grid
            pred = model.predict(Xp)
            ok = np.isfinite(grid) & np.isfinite(pred)
            if ok.sum() > 3 and np.ptp(grid[ok]) > 0:
                slope = float(np.polyfit(grid[ok], pred[ok], 1)[0])
            else:
                slope = np.nan          # бинарный/константный признак — знак не определён
            rows.append({"block": b, "feature": c, "pdp_slope": slope,
                         "pdp_sign": (int(np.sign(slope))
                                      if np.isfinite(slope) else np.nan)})
    return pd.DataFrame(rows)


def ols_traditional(df: pd.DataFrame, target: str = "BMXBMI") -> pd.DataFrame:
    """OLS на стандартизованных признаках + bootstrap 1000 по строкам."""
    cols = [c for f in FEATURES.values() for c in f]
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std(ddof=0)
    X.insert(0, "const", 1.0)
    y = df[target].astype(float).values
    Xv = X.values

    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        b, *_ = np.linalg.lstsq(Xv[idx], y[idx], rcond=None)
        boots.append(b)
    boots = np.array(boots)
    out = pd.DataFrame({
        "feature": X.columns,
        "beta_std": np.round(beta, 4),
        "ci_lo": np.round(np.percentile(boots, 2.5, axis=0), 4),
        "ci_hi": np.round(np.percentile(boots, 97.5, axis=0), 4),
    })
    out["significant"] = (out.ci_lo > 0) | (out.ci_hi < 0)
    block_of = {c: b for b, cs in FEATURES.items() for c in cs}
    out["block"] = out.feature.map(block_of)
    return out


TARGETS = {
    "BMXBMI": "bmi",
    "BPXOSY1": "sbp",
    "BPXOPLS1": "pulse",
}


def main() -> None:
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "BMXBMI"
    suffix = TARGETS.get(target, target.lower())
    t0 = time.perf_counter()
    df = build_cohort().dropna(subset=[target])
    print(f"когорта ({target}): {len(df):,} человек")

    runs = []
    for s in SEEDS:
        r = run_seed(df, s, target=target)
        runs.append(r["blocks"])
        print(f"  seed {s}: R2={r['blocks'].r2_test.iloc[0]:.3f} "
              f"n_test={r['blocks'].n_test.iloc[0]}")
        if s == SEEDS[0]:
            fi = feature_importance(r["model"], r["Xte"], r["yte"], s)
            pdp = pdp_direction(r["model"], r["Xte"], s)

    blocks = pd.concat(runs, ignore_index=True)
    blocks["n_features"] = blocks.block.map({b: len(v) for b, v in FEATURES.items()})
    blocks["delta_r2_per_feature"] = blocks.delta_r2 / blocks.n_features
    blocks.to_csv(OUT / f"pilot_block_importance_{suffix}.csv", index=False,
                  encoding="utf-8-sig")
    fi.to_csv(OUT / f"pilot_feature_importance_{suffix}.csv", index=False,
              encoding="utf-8-sig")
    ols = ols_traditional(df, target)
    ols.to_csv(OUT / f"pilot_ols_{suffix}.csv", index=False, encoding="utf-8-sig")

    # --- H5: пересечение CI лучшего блока с остальными
    first = blocks[blocks.seed == SEEDS[0]].sort_values("delta_r2", ascending=False)
    top = first.iloc[0]
    overlaps = [
        {"block": r.block, "overlap_with_top": bool(
            r.ci_lo <= top.ci_hi and top.ci_lo <= r.ci_hi)}
        for r in first.iloc[1:].itertuples()
    ]

    # --- H7: устойчивость ранга между seed
    piv = blocks.pivot_table(index="block", columns="seed", values="rank")
    stable = bool((piv.nunique(axis=1) == 1).all())

    # --- H8: доля блока у 1-2 ведущих признаков
    h8 = []
    for b in BLOCKS:
        sub = fi[fi.block == b].sort_values("delta_r2", ascending=False)
        tot = sub.delta_r2.sum()
        h8.append({"block": b,
                   "top1": sub.iloc[0].feature,
                   "top1_share": round(float(sub.iloc[0].delta_r2 / tot * 100), 1)
                   if tot > 0 else 0.0,
                   "top2_share": round(float(sub.iloc[:2].delta_r2.sum() / tot * 100), 1)
                   if tot > 0 else 0.0})

    # --- H6: совпадение знаков OLS (традиционный способ) и PDP (ML)
    ols2 = ols.set_index("feature")
    pdp2 = pdp.set_index("feature")
    common = [c for c in pdp2.index
              if c in ols2.index and np.isfinite(pdp2.loc[c, "pdp_sign"])]
    h6 = pd.DataFrame({
        "feature": common,
        "beta_std": [float(ols2.loc[c, "beta_std"]) for c in common],
        "ols_sign": [int(np.sign(ols2.loc[c, "beta_std"])) for c in common],
        "pdp_sign": [int(pdp2.loc[c, "pdp_sign"]) for c in common],
    })
    h6["agree"] = h6.ols_sign == h6.pdp_sign
    h6["significant_ols"] = [bool(ols2.loc[c, "significant"]) for c in common]
    h6.to_csv(OUT / f"pilot_h6_signs_{suffix}.csv", index=False,
              encoding="utf-8-sig")
    agree = int(h6.agree.sum())
    agree_sig = int((h6.agree & h6.significant_ols).sum())
    n_sig = int(h6.significant_ols.sum())

    summary = {
        "cohort_n": int(len(df)),
        "target": target,
        "r2_by_seed": {int(k): float(v) for k, v in
                       blocks.groupby("seed").r2_test.first().items()},
        "top_block": top.block,
        "top_delta_r2": float(top.delta_r2),
        "top_ci": [float(top.ci_lo), float(top.ci_hi)],
        "ci_overlaps": overlaps,
        "h5_non_overlapping": bool(not any(o["overlap_with_top"] for o in overlaps)),
        "rank_by_seed": piv.to_dict(),
        "h7_rank_stable": stable,
        "h8_top1_top2_share": h8,
        "h6_sign_agreement": f"{agree}/{len(common)}",
        "h6_sign_agreement_significant": f"{agree_sig}/{n_sig}",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT / f"pilot_summary_{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(blocks[["block", "seed", "delta_r2", "ci_lo", "ci_hi", "rank",
                  "r2_test"]].to_string(index=False))
    print(f"\nH5 непересекающийся CI: {summary['h5_non_overlapping']} "
          f"(лидер {top.block}, ΔR2={top.delta_r2:.4f})")
    print(f"H7 ранг устойчив: {stable}")
    print(f"H6 совпадение знаков: {summary['h6_sign_agreement']}")
    print(f"время: {summary['seconds']} c")


if __name__ == "__main__":
    main()
