#!/usr/bin/env bash
# Скачивание открытых данных для Темы 1 «Привычки и здоровье».
# Запуск из analysis/t1:  bash scripts/fetch.sh
#
# Все источники открытые:
#   BRFSS 2024      — public domain (CDC)
#   NHANES 2021-23  — public domain (CDC/NCHS)
#   AI4FoodDB       — CC BY-NC-ND 4.0 (см. LICENSE.md), только для НИ, не для прода
#
# Полный AI4FoodDB (100 участников, wearable-ряды) лежит на bidalab.eps.uam.es
# и на 21.11.2026 недоступен (соединение не устанавливается). В GitHub-репозитории
# AI4Food/AI4FoodDB — только пример структуры: 29 CSV, данные одного участника.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p raw processed out

BRFSS_ZIP="https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"
BRFSS_SHA="d33de91a584e866f657d29b1d5b51b5672467286ed22d021e36ee3c7b4d712a4"

# --- 1. BRFSS 2024 (83 МБ zip -> 1,09 ГБ XPT) -------------------------------
if [ ! -f raw/LLCP2024XPT.zip ]; then
  echo "[1/3] BRFSS 2024"
  curl -L --retry 3 -o raw/LLCP2024XPT.zip "$BRFSS_ZIP"
fi
echo "$BRFSS_SHA  raw/LLCP2024XPT.zip" | sha256sum -c -

if [ ! -f raw/LLCP2024.XPT ]; then
  ( cd raw && unzip -o LLCP2024XPT.zip && mv -f "LLCP2024.XPT " LLCP2024.XPT 2>/dev/null || true )
fi

# --- 2. NHANES 2021-2023 (цикл L): 7 модулей, ~18 МБ -------------------------
NH_BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
mkdir -p raw/NHANES_2021_2023
for m in DEMO BMX BPXO SLQ PAQ DR1TOT DPQ; do
  f="raw/NHANES_2021_2023/${m}_L.xpt"
  if [ ! -f "$f" ]; then
    echo "[2/3] NHANES ${m}_L"
    curl -sS -L --retry 2 -o "$f" "$NH_BASE/${m}_L.xpt"
  fi
done

# --- 3. AI4FoodDB: пример структуры из GitHub (полный набор недоступен) ------
# Если полный zip когда-нибудь отдастся, положить его в raw/AI4FoodDB_full.zip
if [ ! -d raw/AI4FoodDB ]; then
  echo "[3/3] AI4FoodDB (GitHub, пример структуры)"
  tmp="$(mktemp -d)"
  curl -sSL --retry 2 -o "$tmp/ai4food.zip" \
    "https://github.com/AI4Food/AI4FoodDB/archive/refs/heads/master.zip"
  unzip -q "$tmp/ai4food.zip" -d "$tmp"
  cp -r "$tmp"/AI4FoodDB-master/datasets raw/AI4FoodDB
  rm -rf "$tmp"
fi

echo
echo "Готово. Дальше:"
echo "  python scripts/convert_brfss.py"
echo "  python scripts/convert_nhanes.py"
echo "  python scripts/convert_ai4food.py"
echo "  python scripts/explore_vars.py"
echo "  python scripts/pilot_importance.py BMXBMI   # или BPXOSY1 / BPXOPLS1"
