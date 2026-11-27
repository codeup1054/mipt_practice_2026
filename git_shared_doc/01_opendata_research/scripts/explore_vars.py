"""Разведка данных под гипотезы H1-H8 (BRFSS 2024 / NHANES 2021-2023 / AI4FoodDB).

Пишет в out/:
  brfss_vars.csv       — наличие, пропуски и коды ключевых колонок BRFSS
  nhanes_vars.csv      — то же для NHANES
  targets_brfss.csv    — распределения целевых показателей BRFSS
  targets_nhanes.csv   — распределения целевых показателей NHANES
  ai4food_inventory.csv— состав локальной копии AI4FoodDB

Запуск:
    PYTHONIOENCODING=utf-8 python scripts/explore_vars.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

BRFSS = ROOT / "processed" / "brfss2024.parquet"
NHANES = ROOT / "processed" / "nhanes"

# --- BRFSS: что ищем под каждую гипотезу -------------------------------
BRFSS_CANDIDATES = {
    # демография / веса
    "_AGEG5YR": ("demo", "возрастная группа 18-80+"),
    "_SEX": ("demo", "пол"),
    "_RACE": ("demo", "раса"),
    "_EDUCAG": ("demo", "образование (4 группы)"),
    "_INCOMG1": ("demo", "доход (5 групп)"),
    "_CLLCPWT": ("demo", "вес выборки (final)"),
    # исходы
    "_BMI5": ("outcome", "ИМТ x100 (кг/м2)"),
    "_BMI5CAT": ("outcome", "категория ИМТ 1-4"),
    "GENHLTH": ("outcome", "самооценка здоровья 1-5"),
    "PHYSHLTH": ("outcome", "дней плохого физ. здоровья"),
    "_PHYS14D": ("outcome", "физ. здоровье: 0-13 дней / 14+ (НЕ активность)"),
    "MENTHLTH": ("outcome", "дней плохого мент. здоровья"),
    "_MENT14D": ("outcome", "мент. здоровье: 0-13 / 14+"),
    "ADDEPEV3": ("outcome", "диагноз депрессии"),
    "DIABETE4": ("outcome", "диабет"),
    # давление и пульс
    "BPHIGH4": ("bp", "гипертония (диагноз)"),
    "BPHIGH": ("bp", "гипертония"),
    "_RFHYPE5": ("bp", "гипертония (derived)"),
    "BPMEDS": ("bp", "приём препаратов от давления"),
    "_RFHYPE6": ("bp", "гипертония (новая версия)"),
    "PULSE": ("pulse", "пульс"),
    # сон
    "SLEPTIM1": ("sleep", "часы сна"),
    "SLEPTIM2": ("sleep", "часы сна (вариант)"),
    "SLEEP": ("sleep", "сон (любая колонка)"),
    # стресс
    "STRESS": ("stress", "стресс"),
    "SDLONELY": ("stress-proxy", "одиночество"),
    "EMTSUPRT": ("stress-proxy", "эмоциональная поддержка"),
    "LSATISFY": ("stress-proxy", "удовлетворённость жизнью"),
    # питание
    "_FRTLT1": ("diet", "фрукты <1 раза/день"),
    "_VEGLT1": ("diet", "овощи <1 раза/день"),
    "_FRTEX1": ("diet", "фрукты/соки"),
    "_VEGEX1": ("diet", "овощи"),
    "_FRUIT": ("diet", "фрукты"),
    "_VEGET": ("diet", "овощи"),
    "SSBSUGR2": ("diet", "сладкие напитки (частота)"),
    "SSBFRUT3": ("diet", "фруктовые напитки (частота)"),
    "WATER": ("diet", "вода"),
    # активность
    "EXERANY2": ("activity", "любая физ. активность, бинарно"),
    "_TOTINDA": ("activity", "активность в свободное время, бинарно"),
    "_PA150R4": ("activity", "≥150 мин/нед"),
    "_PA300R4": ("activity", "≥300 мин/нед"),
    "_PADUR1": ("activity", "минуты активности"),
    # курение / алкоголь
    "SMOKE100": ("smoke", "курил 100 сигарет"),
    "_SMOKER3": ("smoke", "статус курения 1-4"),
    "_RFSMOK3": ("smoke", "курит сейчас"),
    "_DRNKWK3": ("alcohol", "порций в неделю"),
    "_RFDRHV9": ("alcohol", "тяжёлое употребление"),
    "DRNKANY6": ("alcohol", "употреблял алкоголь"),
}

NHANES_CANDIDATES = {
    # DEMO
    "RIDAGEYR": ("DEMO", "возраст"),
    "RIAGENDR": ("DEMO", "пол"),
    "RIDRETH3": ("DEMO", "раса/этничность"),
    "DMDEDUC2": ("DEMO", "образование (взрослые)"),
    "INDFMPIR": ("DEMO", "доход/бедность (PIR)"),
    "WTMEC2YR": ("DEMO", "вес выборки MEC"),
    "SDMVPSU": ("DEMO", "PSU (дизайн)"),
    "SDMVSTRA": ("DEMO", "страта (дизайн)"),
    # BMX
    "BMXBMI": ("BMX", "ИМТ"),
    "BMXWT": ("BMX", "вес"),
    "BMXHT": ("BMX", "рост"),
    "BMXWAIST": ("BMX", "окружность талии"),
    # BPXO (2021-2023: осциллометрия, не BPXSY1!)
    "BPXOSY1": ("BPXO", "САД, 1-е измерение"),
    "BPXODI1": ("BPXO", "ДАД, 1-е измерение"),
    "BPXOPLS1": ("BPXO", "пульс, 1-е измерение"),
    "BPXSY1": ("BPXO", "САД (имя из старых циклов)"),
    "BPXDI1": ("BPXO", "ДАД (имя из старых циклов)"),
    "BPXPLS": ("BPXO", "пульс (имя из старых циклов)"),
    # SLQ
    "SLD012": ("SLQ", "часы сна в будни"),
    "SLD013": ("SLQ", "часы сна в выходные"),
    "SLQ300": ("SLQ", "время засыпания в будни"),
    "SLQ310": ("SLQ", "время подъёма в будни"),
    # PAQ
    "PAD790Q": ("PAQ", "частота интенсивной активности"),
    "PAD790U": ("PAQ", "единица частоты"),
    "PAD800": ("PAQ", "минут интенсивной активности"),
    "PAD810Q": ("PAQ", "частота умеренной активности"),
    "PAD810U": ("PAQ", "единица частоты"),
    "PAD820": ("PAQ", "минут ходьбы/велосипеда"),
    "PAD680": ("PAQ", "минут сидения в день"),
    # DR1TOT
    "DR1DRSTZ": ("DR1TOT", "статус 24h recall (1 = достоверный)"),
    "DR1TKCAL": ("DR1TOT", "ккал"),
    "DR1TPROT": ("DR1TOT", "белок, г"),
    "DR1TCARB": ("DR1TOT", "углеводы, г"),
    "DR1TTFAT": ("DR1TOT", "жиры, г"),
    "DR1TFIBE": ("DR1TOT", "клетчатка, г"),
    "DR1TSODI": ("DR1TOT", "натрий, мг"),
    "DR1TPOTA": ("DR1TOT", "калий, мг"),
    "DR1TVC": ("DR1TOT", "витамин C, мг"),
    "DR1_320Z": ("DR1TOT", "вода, г"),
    "DR1TVEG": ("DR1TOT", "овощи, г (может отсутствовать)"),
    # DPQ (PHQ-9)
    "DPQ010": ("DPQ", "мало интереса/удовольствия"),
    "DPQ020": ("DPQ", "подавленность"),
    "DPQ090": ("DPQ", "суицидальные мысли"),
}


def brfss_summary() -> None:
    schema = pq.read_schema(BRFSS)
    names = set(schema.names)
    wanted = sorted({n for n in BRFSS_CANDIDATES if n in names})
    df = pq.read_table(BRFSS, columns=wanted).to_pandas()

    rows = []
    for var, (block, label) in BRFSS_CANDIDATES.items():
        present = var in names
        rec = {"variable": var, "block": block, "label": label,
               "present": present, "n_nonmissing": "", "pct_nonmissing": "",
               "min": "", "max": "", "top_values": "", "note": ""}
        if present:
            s = df[var]
            if pd.api.types.is_numeric_dtype(s) and not isinstance(
                    s.dtype, pd.CategoricalDtype):
                nn = s.notna()
                rec.update(
                    n_nonmissing=int(nn.sum()),
                    pct_nonmissing=round(nn.mean() * 100, 1),
                    min=float(s.min()) if nn.any() else "",
                    max=float(s.max()) if nn.any() else "",
                    top_values="; ".join(f"{k:g}:{v}" for k, v in
                                         s.value_counts().head(6).items()),
                )
            else:
                rec.update(n_nonmissing=int(s.notna().sum()),
                           pct_nonmissing=round(s.notna().mean() * 100, 1),
                           top_values="; ".join(f"{k}:{v}" for k, v in
                                                s.value_counts().head(6).items()))
        else:
            rec["note"] = "НЕТ КОЛОНКИ в LLCP2024"
        rows.append(rec)

    pd.DataFrame(rows).to_csv(OUT / "brfss_vars.csv", index=False,
                              encoding="utf-8-sig")
    print(f"brfss_vars.csv: {len(rows)} переменных, "
          f"{sum(r['present'] for r in rows)} есть")

    # --- распределения целевых показателей
    cols = [c for c in ["_BMI5", "_BMI5CAT", "GENHLTH", "DIABETE4",
                        "_MENT14D", "_PHYS14D", "_SMOKER3", "_DRNKWK3",
                        "EXERANY2", "_TOTINDA", "SSBSUGR2"] if c in names]
    t = pq.read_table(BRFSS, columns=cols).to_pandas()

    out = []
    bmi = t["_BMI5"] / 100
    out.append({"target": "BMI (kg/m2)", "n": int(bmi.notna().sum()),
                "mean": round(bmi.mean(), 2), "sd": round(bmi.std(), 2),
                "p5": round(bmi.quantile(.05), 2), "median": round(bmi.median(), 2),
                "p95": round(bmi.quantile(.95), 2), "max": round(bmi.max(), 2),
                "note": "_BMI5 / 100; коды 9999 -> пропуск"})
    for col, lab in [("GENHLTH", "GENHLTH 1=отл..5=плох"),
                     ("_BMI5CAT", "_BMI5CAT 1=нед..4=ожир"),
                     ("_MENT14D", "_MENT14D"),
                     ("_PHYS14D", "_PHYS14D"),
                     ("DIABETE4", "DIABETE4"),
                     ("EXERANY2", "EXERANY2"),
                     ("_TOTINDA", "_TOTINDA"),
                     ("_SMOKER3", "_SMOKER3")]:
        vc = t[col].value_counts(dropna=False).sort_index()
        out.append({"target": lab, "n": int(t[col].notna().sum()),
                    "mean": "", "sd": "", "p5": "", "median": "",
                    "p95": "", "max": "",
                    "note": "; ".join(f"{k:g}:{v}" for k, v in vc.items())})
    pd.DataFrame(out).to_csv(OUT / "targets_brfss.csv", index=False,
                             encoding="utf-8-sig")
    print("targets_brfss.csv готов")


def nhanes_summary() -> None:
    frames = {}
    present = {}
    for mod in ["DEMO", "BMX", "BPXO", "SLQ", "PAQ", "DR1TOT", "DPQ"]:
        p = NHANES / f"{mod}_L.parquet"
        if p.exists():
            frames[mod] = pq.read_table(p).to_pandas()
            present[mod] = set(frames[mod].columns)

    rows = []
    for var, (mod, label) in NHANES_CANDIDATES.items():
        ok = var in present.get(mod, set())
        rec = {"file": f"{mod}_L", "variable": var, "label": label,
               "present": ok, "n_rows_file": len(frames.get(mod, [])),
               "n_nonmissing": "", "pct_nonmissing": "", "min": "", "max": "",
               "top_values": "", "caution": ""}
        if ok:
            s = frames[mod][var]
            nn = s.notna()
            numeric = (pd.api.types.is_numeric_dtype(s)
                       and not isinstance(s.dtype, pd.CategoricalDtype))
            rec.update(n_nonmissing=int(nn.sum()),
                       pct_nonmissing=round(nn.mean() * 100, 1),
                       min=float(s.min()) if numeric and nn.any() else "",
                       max=float(s.max()) if numeric and nn.any() else "",
                       top_values="; ".join(f"{k:g}:{v}" for k, v in
                                            s.value_counts().head(6).items())
                       if numeric else
                       "; ".join(f"{k}:{v}" for k, v in
                                 s.value_counts().head(6).items()))
            if var in {"BPXSY1", "BPXDI1", "BPXPLS"}:
                rec["caution"] = "в 2021-2023 таких имён нет, см. BPXO*"
        else:
            rec["caution"] = "нет колонки в модуле"
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "nhanes_vars.csv", index=False,
                              encoding="utf-8-sig")
    print(f"nhanes_vars.csv: {sum(r['present'] for r in rows)}/{len(rows)} есть")

    # --- целевые показатели
    demo = frames["DEMO"][["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bmx = frames["BMX"][["SEQN", "BMXBMI", "BMXWAIST"]]
    bp = frames["BPXO"][["SEQN", "BPXOSY1", "BPXODI1", "BPXOPLS1"]]
    base = demo.merge(bmx, on="SEQN", how="left").merge(bp, on="SEQN", how="left")
    adults = base[base.RIDAGEYR >= 20]

    out = []
    for col, lab in [("BMXBMI", "ИМТ, кг/м2"), ("BMXWAIST", "талия, см"),
                     ("BPXOSY1", "САД, мм рт.ст."), ("BPXODI1", "ДАД, мм рт.ст."),
                     ("BPXOPLS1", "пульс, уд/мин")]:
        s = adults[col].dropna()
        out.append({"metric": lab, "col": col, "n": int(len(s)),
                    "mean": round(s.mean(), 1), "sd": round(s.std(), 1),
                    "p5": round(s.quantile(.05), 1),
                    "median": round(s.median(), 1),
                    "p95": round(s.quantile(.95), 1)})
    pd.DataFrame(out).to_csv(OUT / "targets_nhanes.csv", index=False,
                             encoding="utf-8-sig")

    # доля гипертонии по порогу 130/80 (ACC/AHA)
    a = adults.dropna(subset=["BPXOSY1", "BPXODI1"])
    htn = ((a.BPXOSY1 >= 130) | (a.BPXODI1 >= 80)).mean() * 100
    print(f"NHANES adults 20+: n={len(adults)}, АГ≥130/80 = {htn:.1f}%")


def ai4food_inventory() -> None:
    meta = json.loads((OUT / "ai4fooddb_convert.json").read_text(encoding="utf-8"))
    rows = [{"file": d["file"], "rows": d["rows"], "cols": d["cols"],
             "src_kb": round(d["src_bytes"] / 1024, 1),
             "parquet_kb": round(d["parquet_bytes"] / 1024, 1)}
            for d in meta["datasets"]]
    # сколько уникальных участников в каждом файле
    for r in rows:
        df = pd.read_csv(ROOT / "raw" / "AI4FoodDB" / r["file"], low_memory=False)
        if "id" in df.columns:
            r["participants"] = df["id"].nunique()
        else:
            r["participants"] = ""
    pd.DataFrame(rows).to_csv(OUT / "ai4food_inventory.csv", index=False,
                              encoding="utf-8-sig")
    print("ai4food_inventory.csv готов")


if __name__ == "__main__":
    brfss_summary()
    nhanes_summary()
    ai4food_inventory()
