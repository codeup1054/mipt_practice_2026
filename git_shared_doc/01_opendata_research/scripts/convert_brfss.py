"""BRFSS 2024 (LLCP2024.XPT) -> Parquet.

Читает SAS XPORT чанками, приводит типы (float32 для числовых, category для
строковых) и пишет Parquet в processed/.

Запуск:
    PYTHONIOENCODING=utf-8 python scripts/convert_brfss.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw" / "LLCP2024.XPT"
DST = ROOT / "processed" / "brfss2024.parquet"
META = ROOT / "out" / "brfss2024_convert.json"

CHUNK = 100_000


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
            # BRFSS XPT приходит как object/bytes -> строки; в Parquet уйдут
            # словарём (dictionary<...>) — см. to_dictionary()
            out[col] = s.astype("string")
    return pd.DataFrame(out, index=df.index)


DICT = pa.dictionary(pa.int32(), pa.string())


def to_dictionary(table: pa.Table) -> pa.Table:
    """Строковые колонки -> dictionary<int32,string> с одинаковым типом в чанках."""
    fields = [
        pa.field(f.name, DICT if pa.types.is_string(f.type) else f.type)
        for f in table.schema
    ]
    return table.cast(pa.schema(fields))


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    reader = pd.read_sas(SRC, format="xport", chunksize=CHUNK, encoding="utf-8")

    writer: pq.ParquetWriter | None = None
    rows = 0
    cols = 0
    schema: pa.Schema | None = None

    for i, chunk in enumerate(reader):
        chunk = cast(chunk)
        table = to_dictionary(pa.Table.from_pandas(chunk, preserve_index=False))
        if writer is None:
            # фиксируем схему по первому чанку: у category индекс может быть
            # int8/int16/int32 в зависимости от числа уровней, приводим к одной
            schema = table.schema
            writer = pq.ParquetWriter(DST, schema, compression="zstd")
        else:
            table = table.cast(schema)
        writer.write_table(table)
        rows += len(chunk)
        cols = chunk.shape[1]
        print(f"  chunk {i}: +{len(chunk):,} rows (total {rows:,})", flush=True)

    if writer is not None:
        writer.close()

    elapsed = time.perf_counter() - t0
    size_src = SRC.stat().st_size
    size_dst = DST.stat().st_size

    # контрольное чтение
    t1 = time.perf_counter()
    check = pq.read_table(DST, columns=["_BMI5", "GENHLTH", "SSBSUGR2"]).to_pandas()
    read_time = time.perf_counter() - t1

    meta = {
        "source": str(SRC),
        "source_bytes": size_src,
        "parquet": str(DST),
        "parquet_bytes": size_dst,
        "compression": "zstd",
        "rows": rows,
        "columns": cols,
        "convert_seconds": round(elapsed, 1),
        "read_3cols_seconds": round(read_time, 2),
        "dtypes": {f.name: str(f.type) for f in (schema or [])},
        "sample_rows": int(len(check)),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nГотово: {DST}")
    print(f"строк={rows:,} колонок={cols}  XPT={size_src / 1e6:.1f} МБ  "
          f"parquet={size_dst / 1e6:.1f} МБ  время={elapsed:.1f} c")


if __name__ == "__main__":
    main()
