"""
Popularity candidates: внутри категорий пользователя + глобальная популярность.
"""
import polars as pl
from dataclasses import dataclass
from .utils.io import scan_parquet
from .utils.logging import setup_logger

@dataclass
class PopConfig:
    user_id: str
    item_id: str
    ts: str
    category_id: str
    per_user_from_pop: int
    val_start_ts: int
    history_window_days: int

def build_pop_candidates(cfg, interactions_path: str, items_path: str, val_start_ts: int) -> pl.DataFrame:
    log = setup_logger("pop")
    lf = scan_parquet(interactions_path).filter(pl.col(cfg.ts) < val_start_ts)
    window_start = val_start_ts - cfg.history_window_days * 24 * 3600 * 1000
    lf = lf.filter(pl.col(cfg.ts) >= window_start)

    items = scan_parquet(items_path).select([cfg.item_id, cfg.category_id])

    # топ категорий пользователя
    user_top_cats = (
        lf.join(items, on=cfg.item_id)
          .group_by([cfg.user_id, cfg.category_id])
          .count()
          .sort("count", descending=True)
          .group_by(cfg.user_id).head(3)
    )

    # популярные товары в этих категориях
    global_pop = (
        lf.group_by(cfg.item_id).count().rename({"count":"pop"})
        .join(items, on=cfg.item_id)
    )

    joined = user_top_cats.join(global_pop, on=cfg.category_id).select([cfg.user_id, cfg.item_id, "pop"])
    candidates = (
        joined.sort(["user_id","pop"], descending=[False, True])
              .group_by(cfg.user_id).head(cfg.per_user_from_pop)
              .with_columns([pl.lit("pop").alias("source")])
    )
    log.info("Popularity candidates ready")
    return candidates
