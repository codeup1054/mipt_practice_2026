"""AI4FoodDB CSV -> Parquet.

Локальная копия — пример структуры (каталог datasets из GitHub AI4Food/AI4FoodDB):
29 CSV, DS1-DS10, но данные только одного участника A4F_10021.
Конвертируем каждый CSV отдельно, чтобы сохранить разбиение по датасетам.

Запуск:
    PYTHONIOENCODING=utf-8 python scripts/convert_ai4food.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "raw" / "AI4FoodDB"
DST = ROOT / "processed" / "ai4fooddb"
META = ROOT / "out" / "ai4fooddb_convert.json"


def cast(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("float32")
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("bool")
        else:
            # id, category, timestamp и прочие справочники -> строки + словарь
            out[col] = s.astype("string")
    return pd.DataFrame(out)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.rglob("*.csv"))
    report = []
    total_src = 0
    total_dst = 0
    t_all = time.perf_counter()

    for f in files:
        rel = f.relative_to(SRC)
        t0 = time.perf_counter()
        df = pd.read_csv(f, low_memory=False)
        t_read = time.perf_counter() - t0

        table = pa.Table.from_pandas(cast(df), preserve_index=False)
        # строковые -> dictionary<int32,string>, чтобы типы не зависели от чанка
        fields = [
            pa.field(x.name, pa.dictionary(pa.int32(), pa.string())
                     if pa.types.is_string(x.type) else x.type)
            for x in table.schema
        ]
        table = table.cast(pa.schema(fields))

        out = DST / (rel.with_suffix("").as_posix().replace("/", "__") + ".parquet")
        pq.write_table(table, out, compression="zstd")

        src_b = f.stat().st_size
        dst_b = out.stat().st_size
        total_src += src_b
        total_dst += dst_b

        t1 = time.perf_counter()
        pq.read_table(out).to_pandas()
        t_pq = time.perf_counter() - t1

        report.append({
            "file": rel.as_posix(),
            "parquet": out.name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "src_bytes": src_b,
            "parquet_bytes": dst_b,
            "read_csv_seconds": round(t_read, 3),
            "read_parquet_seconds": round(t_pq, 3),
            "columns": list(df.columns),
        })
        print(f"{rel.as_posix():75s} {df.shape[0]:>7}x{df.shape[1]:<3} "
              f"{src_b/1024:8.0f} KB -> {dst_b/1024:7.0f} KB")

    meta = {
        "source_dir": str(SRC),
        "files": len(files),
        "total_src_bytes": total_src,
        "total_parquet_bytes": total_dst,
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "datasets": report,
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nCSV={total_src/1e6:.2f} МБ -> parquet={total_dst/1e6:.2f} МБ, "
          f"файлов={len(files)}")


if __name__ == "__main__":
    main()
