"""
Co-visitation: item->item по окну истории с весами событий и экспон. затуханием.
Выдаёт кандидатов для каждого пользователя на основе его последних N взаимодействий.
"""
from dataclasses import dataclass
import polars as pl
import numpy as np
from .utils.io import scan_parquet
from .utils.logging import setup_logger

@dataclass
class CoVisConfig:
    user_id: str
    item_id: str
    event_type: str
    ts: str
    events_weights: dict
    decay_half_life_days: int
    history_window_days: int
    topk_per_anchor: int
    per_user_from_covis: int

def _decay_weight(ts_now: int, ts_event: int, half_life_ms: int) -> float:
    dt = max(0, ts_now - ts_event)
    return 0.5 ** (dt / half_life_ms)

def build_covis_candidates(cfg, interactions_path: str, val_start_ts: int) -> pl.DataFrame:
    log = setup_logger("covis")
    lf = scan_parquet(interactions_path).filter(
        pl.col(cfg.ts) < val_start_ts
    )
    # берем последние history_window_days для каждого пользователя
    # упрощенно: просто фильтр по ts > val_start - window
    window_start = val_start_ts - cfg.history_window_days * 24 * 3600 * 1000
    lf = lf.filter(pl.col(cfg.ts) >= window_start)

    # для простоты: co-vis как совместные появления item в истории пользователя (self-join)
    base = lf.select([cfg.user_id, cfg.item_id, cfg.event_type, cfg.ts])
    left = base.rename({cfg.item_id: "anchor_item", cfg.ts: "ts_left", cfg.event_type: "evt_left"})
    right = base.rename({cfg.item_id: "cand_item", cfg.ts: "ts_right", cfg.event_type: "evt_right"})

    pairs = left.join(right, on=cfg.user_id)
    pairs = pairs.filter(pl.col("anchor_item") != pl.col("cand_item"))

    # вес = event_weight_left * event_weight_right * decay
    ew = cfg.events_weights
    pairs = pairs.with_columns([
        pl.col("evt_left").map_dict(ew, default=1.0).alias("w_left"),
        pl.col("evt_right").map_dict(ew, default=1.0).alias("w_right"),
    ])
    half_life_ms = cfg.decay_half_life_days * 24 * 3600 * 1000
    pairs = pairs.with_columns([
        pl.struct(["ts_left","ts_right"]).map_elements(
            lambda s: _decay_weight(
                ts_now=val_start_ts, 
                ts_event=int(0.5*(s["ts_left"]+s["ts_right"]))
            , half_life_ms=half_life_ms)
        ).alias("w_decay")
    ])
    pairs = pairs.with_columns((pl.col("w_left")*pl.col("w_right")*pl.col("w_decay")).alias("w"))

    covis = pairs.group_by(["anchor_item","cand_item"]).agg(pl.col("w").sum().alias("score"))

    # topK для каждого anchor_item
    covis_top = (
        covis.sort("score", descending=True)
             .group_by("anchor_item")
             .head(cfg.topk_per_anchor)
    )

    # теперь превратим в кандидатов per user: берём последние anchor_item пользователя и мержим
    last_items = (
        base.group_by(cfg.user_id)
            .agg(pl.all().sort_by(cfg.ts, descending=True))
            .select([cfg.user_id, pl.col(cfg.item_id).arr.head(20).alias("recent_items")])
    )
    exploded = last_items.explode("recent_items").rename({"recent_items": "anchor_item"})
    user_cands = exploded.join(covis_top, on="anchor_item").select([cfg.user_id, pl.col("cand_item").alias("item_id"), pl.col("score")])

    # суммируем по item_id для юзера, берём top-N
    user_cands = (
        user_cands.group_by([cfg.user_id, "item_id"]).agg(pl.col("score").sum())
        .sort(["user_id", "score"], descending=[False, True])
        .group_by(cfg.user_id).head(cfg.per_user_from_covis)
        .with_columns([
            pl.lit("covis").alias("source")
        ])
    )
    log.info("CoVis candidates ready")
    return user_cands
