import argparse, polars as pl, pandas as pd, numpy as np
from .config import load_config
from .utils.logging import setup_logger
from .utils.io import scan_parquet

def build_features(cfg, stage: str):
    log = setup_logger("features")
    col = cfg["columns"]; paths = cfg["paths"]

    if stage == "val":
        cands_path = f"{paths['out_dir_interim']}/candidates_val.parquet"
        split_start = pd.Timestamp(cfg["time"]["val_start"]).timestamp()*1000
        split_end   = pd.Timestamp(cfg["time"]["val_end"]).timestamp()*1000
    elif stage == "test":
        # на тест можно реюзать ту же логику генерации кандидатов с другими окнами
        cands_path = f"{paths['out_dir_interim']}/candidates_test.parquet"
        split_start = pd.Timestamp(cfg["time"]["test_start"]).timestamp()*1000
        split_end   = pd.Timestamp(cfg["time"]["test_end"]).timestamp()*1000
    else:
        raise ValueError("stage must be 'val' or 'test'")

    cands = scan_parquet(cands_path)
    interactions = scan_parquet(paths["interactions"]).filter(pl.col(col["ts"]) < split_start)
    items = scan_parquet(paths["items"]).select([col["item_id"], col["category_id"], col["brand"], col["price"]])

    # агрегаты пользователя
    user_stats = (interactions
        .group_by(col["user_id"])
        .agg([
            pl.count().alias("u_events"),
            pl.col(col["ts"]).max().alias("u_last_ts")
        ])
    )

    # popularities (до split)
    item_pop = (
        interactions.group_by(col["item_id"]).count().rename({"count":"i_pop"})
    )

    # join всё к кандидатам
    feats = (cands
        .join(user_stats, on=col["user_id"], how="left")
        .join(item_pop, on=col["item_id"], how="left")
        .join(items, on=col["item_id"], how="left")
        .with_columns([
            pl.col("i_pop").fill_null(0),
            pl.col("u_events").fill_null(0),
        ])
    )

    # цель (для val): куплен в [val_start, val_end]?
    labels = (scan_parquet(paths["orders"])
        .filter((pl.col(col["ts"])>=split_start) & (pl.col(col["ts"])<split_end))
        .filter(pl.col(cfg["columns"]["delivered_flag"])==1)
        .select([col["user_id"], col["item_id"]])
        .with_columns([pl.lit(1).alias("label")])
        .unique()
    )

    if stage == "val":
        feats = feats.join(labels, on=[col["user_id"], col["item_id"]], how="left").with_columns([
            pl.col("label").fill_null(0)
        ])
        out_path = f"{paths['out_dir_processed']}/features_val.parquet"
    else:
        out_path = f"{paths['out_dir_processed']}/features_test.parquet"

    feats.collect().write_parquet(out_path)
    log.info(f"Saved features: {out_path}")

def main(config_path: str, stage: str):
    cfg = load_config(config_path).raw
    build_features(cfg, stage)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["val","test"], required=True)
    args = ap.parse_args()
    main(args.config, args.stage)
