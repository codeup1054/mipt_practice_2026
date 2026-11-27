"""NHANES 2021-2023 (цикл L) XPT -> Parquet.

Модули: DEMO (демография), BMX (антропометрия), BPXO (давление и пульс),
SLQ (сон), PAQ (физическая активность), DR1TOT (24h recall, нутриенты),
DPQ (депрессия/PHQ-9). Все — public domain, wwwn.cdc.gov/Nchs/Nhanes.

Запуск:
    PYTHONIOENCODING=utf-8 python scripts/convert_nhanes.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw" / "NHANES_2021_2023"
DST = ROOT / "processed" / "nhanes"
META = ROOT / "out" / "nhanes_convert.json"


def cast(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            out[col] = s.astype("float32")
        elif pd.api.types.is_integer_dtype(s):
            out[col] = s.astype("int32")
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("bool")
        else:
            out[col] = s.astype("string")
    return pd.DataFrame(out)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("*.xpt"))
    report = []
    for f in files:
        t0 = time.perf_counter()
        df = pd.read_sas(f, format="xport", encoding="utf-8")
        t_read = time.perf_counter() - t0

        table = pa.Table.from_pandas(cast(df), preserve_index=False)
        fields = [
            pa.field(x.name, pa.dictionary(pa.int32(), pa.string())
                     if pa.types.is_string(x.type) else x.type)
            for x in table.schema
        ]
        table = table.cast(pa.schema(fields))

        out = DST / (f.stem + ".parquet")
        pq.write_table(table, out, compression="zstd")

        t1 = time.perf_counter()
        pq.read_table(out).to_pandas()
        t_pq = time.perf_counter() - t1

        report.append({
            "file": f.name,
            "parquet": out.name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "src_bytes": f.stat().st_size,
            "parquet_bytes": out.stat().st_size,
            "read_xpt_seconds": round(t_read, 3),
            "read_parquet_seconds": round(t_pq, 3),
            "columns": list(df.columns),
        })
        print(f"{f.name:14s} {df.shape[0]:>7}x{df.shape[1]:<4} "
              f"{f.stat().st_size/1024:8.0f} KB -> {out.stat().st_size/1024:7.0f} KB "
              f"({t_read:.2f}s -> {t_pq:.2f}s)")

    meta = {
        "cycle": "2021-2023 (L)",
        "source_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/",
        "license": "public domain (US CDC/NCHS)",
        "files": len(files),
        "total_src_bytes": sum(r["src_bytes"] for r in report),
        "total_parquet_bytes": sum(r["parquet_bytes"] for r in report),
        "modules": report,
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nВсего: {meta['total_src_bytes']/1e6:.1f} МБ XPT -> "
          f"{meta['total_parquet_bytes']/1e6:.1f} МБ parquet")


if __name__ == "__main__":
    main()
