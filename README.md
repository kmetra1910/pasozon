# Ozon RecSys — Trajectory B

Pipeline: candidate generation (co-visitation + popularity) → feature engineering → LightGBM ranker (lambdarank) → RRF blend → diversity rerank → submission.

## Логика системы

### Траектория A — базовый baseline без ML

Идея: собрать несколько независимых списков кандидатов и детерминированно склеить их Reciprocal Rank Fusion (RRF) или взвешенной суммой.

- Co-visitation item→item по последним 30–60 дням событий (просмотры/добавления/покупки), с весами: покупка > корзина > просмотр; экспоненциальный decay по времени.
- Популярность в окне (неделя/месяц) внутри категории + фильтр уже купленного.
- Категорийная близость: товары из той же ветки дерева (LCA‑глубина ≥ k).
- (Опционально) CLIP‑соседи для последних N просмотренных/купленных: top‑K ближайших по cosine от готовых эмбеддингов (Faiss/Annoy).
- Слияние списков в 300–1000 кандидатов и RRF‑склейка → топ‑100.
- Diversify: штраф за одинаковые бренды/категории, чтобы не завалить список клон‑товарами.

Плюсы: минимум кода и зависимости, очень стабильно; легко пройти воспроизводимость.  
Минусы: без обучения ранкера потолок качества ниже, чем у B.

### Траектория B — кандидаты из A + LightGBM ранкер

Идея: генераторы из A дают recall, а легковесный ранкер (LightGBM ranker) оптимизирует NDCG@100.

Фичи (минимальный, но «бьющий» набор):

- co‑vis score, позиция в co‑vis списке;
- поп‑счёт в окне (и нормированный внутри категории);
- пересечение по дереву категорий (глубина LCA, расстояние);
- cosine(средний user‑вектор последних M товаров, item_CLIP);
- recency пользователя (время с последнего события), частоты событий;
- базовые товарные: цена/бренд/категория (one‑hot/target mean по времени).

Тренировка: temporal split (train→val), цель — «куплен=1» в валидационном окне, негативы — оставшиеся кандидаты. Objective: lambdarank, metric: ndcg@100.  
Склейка: предсказание ранкера + (слегка) RRF с co‑vis/pop — даёт устойчивость.  
Плюсы: лучший «возврат качества / сложность».  
Минусы: нужно аккуратно не допустить утечек по времени и сделать генерацию негативов.

## Структура модулей

- `recsys/covis.py` — co‑visitation кандидаты.
- `recsys/popularity.py` — популярность в категориях.
- `recsys/build_candidates.py` — сборка и объединение кандидатов.
- `recsys/features.py` — генерация признаков для ранкера.
- `recsys/train_lgbm_ranker.py` — обучение LightGBM ранкера.
- `recsys/blend_rrf.py` — RRF‑склейка со скором ранкера.
- `recsys/rerank_diversity.py` — diversity rerank (MMR/бренды/категории).
- `recsys/make_submission.py` — финальная выдача сабмита.
- `recsys/utils/*` — вспомогательные функции (логгер, IO, seed).

## Quickstart
```bash
poetry install --no-root
poetry run python -m recsys.build_candidates --config config/config.yaml
poetry run python -m recsys.features --config config/config.yaml --stage val
poetry run python -m recsys.train_lgbm_ranker --config config/config.yaml
poetry run python -m recsys.blend_rrf --config config/config.yaml --stage val
poetry run python -m recsys.rerank_diversity --config config/config.yaml --stage val
poetry run python -m recsys.make_submission --config config/config.yaml --stage test
```

Все пути/колонки и temporal split настраиваются в config/config.yaml.

#### `config/config.yaml`
```yaml
paths:
  interactions: "data/raw/tracker/*.parquet"      # user-item events (views, add_to_cart, fav, purchase)
  orders: "data/raw/orders/*.parquet"             # delivered/returned
  items: "data/raw/items/*.parquet"               # catalog + attrs + clip
  categories: "data/raw/categories/*.parquet"
  test_users: "data/raw/test_users/*.parquet"
  out_dir_interim: "data/interim"
  out_dir_processed: "data/processed"
  models_dir: "models"

columns:
  user_id: "user_id"
  item_id: "item_id"
  event_type: "event_type"        # e.g. view/cart/fav/purchase
  ts: "ts"                        # timestamp (int or datetime)
  delivered_flag: "delivered"     # in orders
  category_id: "category_id"
  brand: "brand"
  price: "price"
  clip: "clip_embedding"          # list/array[512]

events_weights:
  purchase: 3.0
  cart: 1.5
  fav: 1.2
  view: 1.0

time:
  # укажите реальные границы по данным
  train_start: "2024-01-01"
  train_end:   "2024-05-31"
  val_start:   "2024-06-01"
  val_end:     "2024-06-15"
  test_start:  "2024-06-16"
  test_end:    "2024-06-30"
  decay_half_life_days: 30

candidates:
  per_user_from_covis: 800
  per_user_from_pop: 500
  topk_per_anchor: 50
  history_window_days: 60

ranker:
  params:
    objective: "lambdarank"
    metric: "ndcg"
    eval_at: [100]
    learning_rate: 0.05
    num_leaves: 127
    min_data_in_leaf: 50
    feature_fraction: 0.8
    lambda_l1: 0.0
    lambda_l2: 0.0
    max_depth: -1
    random_state: 42

blend:
  rrf_k: 60     # RRF hyperparam

diversity:
  max_per_brand: 10
  max_per_category: 20
  mmr_lambda: 0.8  # 1.0 = только relevance, 0.0 = только diversity

submission:
  topk: 100
  filename_val: "data/processed/submission_val.csv"
  filename_test: "data/processed/submission_test.csv"
```
