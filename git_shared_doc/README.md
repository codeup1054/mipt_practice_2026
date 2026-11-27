# Документы для студентов

Набор для самостоятельной работы: что открыть, что запустить и что забрать в портфолио.
Благодарственные письма сюда не входят.

## С чего начать

1. **Портфолио и резюме** — `portfolio/STUDENT_PORTFOLIO_MEMO.html` · `STUDENT_PORTFOLIO_MEMO.md`.
   Что предъявить: команда навыков (skills), OpenCode, экономика токенов
   (token economics), научный метод, хранилище кода (repository), готовые формулировки.
2. **Гид T1** — `01_opendata_research/T1-MEMO_NOTEBOOK_GUIDE.html` · `T1-MEMO_NOTEBOOK_GUIDE.md`.
   12 шагов исследования на данных о здоровье.
3. **Гид T2** — `02_nutrivision/T2-MEMO_NOTEBOOK_GUIDE.html` · `T2-MEMO_NOTEBOOK_GUIDE.md`.
   9 шагов нутриентного анализа: навык (skill), прогоны, MAE %, критерий Стьюдента
   (t-test), вывод.
4. **Ноутбуки** — `01_opendata_research/` и `02_nutrivision/`. Гиды и ноутбуки читаются вместе.

## Что где лежит

| Папка | Содержимое |
|---|---|
| `portfolio/` | памятка «что предъявить в резюме и портфолио»: HTML и Markdown |
| `01_opendata_research/` | `REFERENCE_ANALYSIS.ipynb`, гид, `scripts/`, `out/` с графиками, `skills/` — 10 примеров |
| `02_nutrivision/` | `NUT02_REFERENCE.ipynb`, гид, каталог блюд, фото, навык (skill), `spec.md`, метрики, графики, `skills/` — 10 примеров |

## 01_opendata_research · T1-MEMO · Привычки и здоровье

Ноутбук и графики включены, **сырые данные — нет**: BRFSS, NHANES и AI4FoodDB
весят больше гигабайта и у каждого источника своя лицензия. Как скачать и
сконвертировать — в `01_opendata_research/README.md` и `01_opendata_research/scripts/`. Примеры навыков — `01_opendata_research/skills/`.

Если открываете ноутбук без данных, он дойдёт до проверки источников и остановится:
это эталон, а не готовый датасет.

## 02_nutrivision · T2-MEMO · Нутриентный анализ

Ноутбук самодостаточен: в `02_nutrivision/processed/` лежат каталог из 8 блюд, паспорт и фото.
В `02_nutrivision/out/` — метрики, факторы и шесть графиков.

```bash
cd 02_nutrivision
jupyter execute NUT02_REFERENCE.ipynb
```

Если реальных прогонов нет, ноутбук собирает демонстрационные ответы и помечает
`runs_source: "demo"`. Для сдачи нужны минимум две модели и два чата. Примеры навыков — `02_nutrivision/skills/`.

## Ноутбуки и материалы

- `01_opendata_research/REFERENCE_ANALYSIS.ipynb` — исследование: паспорт, QC, модель, CI, метаанализ, кластеры.
- `02_nutrivision/NUT02_REFERENCE.ipynb` — навык (skill), 2 модели × 2 чата, MAE %, Match %,
  критерий Стьюдента (t-test),
  тепловая карта ошибок, фото блюд.
- `01_opendata_research/out/figures/` и `02_nutrivision/out/figures/` — готовые графики для отчёта и портфолио.
- `02_nutrivision/processed/photos/` — фото блюд; источники и лицензии — в `CREDITS.md`.

## Правила

- не публиковать ключи, `.env` и персональные данные;
- в отчётах ссылаться на числа из `metrics.json`, а не набирать вручную;
- чужие фото использовать только по лицензии;
- если результат ещё не принят, писать «участвовал в разработке», а не «сделал».
