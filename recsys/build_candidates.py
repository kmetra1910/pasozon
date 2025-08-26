import argparse, polars as pl, pandas as pd
from .config import load_config
from .utils.logging import setup_logger
from .covis import CoVisConfig, build_covis_candidates
from .popularity import PopConfig, build_pop_candidates

def main(config_path: str):
    log = setup_logger("build_candidates")
    cfg = load_config(config_path).raw

    col = cfg["columns"]; paths = cfg["paths"]
    time_cfg = cfg["time"]
    # предполагаем, что ts — ms epoch; если datetime — конвертируйте заранее
    import datetime as dt
    val_start_ts = int(pd.Timestamp(time_cfg["val_start"]).timestamp()*1000)

    cov = CoVisConfig(
        user_id=col["user_id"], item_id=col["item_id"], event_type=col["event_type"], ts=col["ts"],
        events_weights=cfg["events_weights"],
        decay_half_life_days=time_cfg["decay_half_life_days"],
        history_window_days=cfg["candidates"]["history_window_days"],
        topk_per_anchor=cfg["candidates"]["topk_per_anchor"],
        per_user_from_covis=cfg["candidates"]["per_user_from_covis"],
    )
    pop = PopConfig(
        user_id=col["user_id"], item_id=col["item_id"], ts=col["ts"], category_id=col["category_id"],
        per_user_from_pop=cfg["candidates"]["per_user_from_pop"],
        val_start_ts=val_start_ts,
        history_window_days=cfg["candidates"]["history_window_days"],
    )

    cov_df = build_covis_candidates(cov, paths["interactions"], val_start_ts)
    pop_df = build_pop_candidates(pop, paths["interactions"], paths["items"], val_start_ts)

    # merge & dedup
    all_df = pl.concat([cov_df, pop_df], how="diagonal")
    all_df = (
        all_df.group_by([col["user_id"], "item_id"])
              .agg(pl.col("score").max(), pl.col("source").first())
              .rename({"score":"cand_score"})
              .sort([col["user_id"], "cand_score"], descending=[False, True])
    )
    out = f"{cfg['paths']['out_dir_interim']}/candidates_val.parquet"
    all_df.collect().write_parquet(out)
    log.info(f"Saved candidates: {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
