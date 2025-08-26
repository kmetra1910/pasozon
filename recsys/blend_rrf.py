import argparse, polars as pl
from .config import load_config
from .utils.logging import setup_logger

def rrf_rank(score_series, k: int = 60):
    # для RRF нам нужны порядковые ранги; тут предполагаем предварительную сортировку
    return 1.0 / (k + score_series.rank("dense", descending=True))

def main(config_path: str, stage: str):
    log = setup_logger("blend_rrf")
    cfg = load_config(config_path).raw
    col = cfg["columns"]; paths = cfg["paths"]; rrf_k = cfg["blend"]["rrf_k"]

    cands_path = f"{paths['out_dir_interim']}/candidates_{stage}.parquet"
    scores_path = f"{paths['out_dir_processed']}/scores_{stage}.parquet" if stage=="val" else f"{paths['out_dir_processed']}/scores_{stage}.parquet"

    cands = pl.read_parquet(cands_path)
    # нормируем cand_score в ранги по пользователю
    cands = cands.with_columns([
        pl.col("cand_score").rank("dense", descending=True).over(col["user_id"]).alias("cand_rank")
    ])
    cands = cands.with_columns([
        (1.0/(rrf_k + pl.col("cand_rank"))).alias("rrf_cand")
    ])

    try:
        scores = pl.read_parquet(scores_path).rename({"lgbm_score":"lgbm_score"})
        scores = scores.with_columns([
            pl.col("lgbm_score").rank("dense", descending=True).over(col["user_id"]).alias("lgbm_rank"),
            (1.0/(rrf_k + pl.col("lgbm_rank"))).alias("rrf_lgbm")
        ])
        blend = cands.join(scores, on=[col["user_id"], col["item_id"]], how="left").with_columns([
            (pl.col("rrf_cand") + pl.col("rrf_lgbm").fill_null(0.0)).alias("blend_score")
        ])
    except FileNotFoundError:
        log.warning("No LGBM scores found, using only candidate RRF")
        blend = cands.with_columns(pl.col("rrf_cand").alias("blend_score"))

    out = f"{paths['out_dir_processed']}/blended_{stage}.parquet"
    blend.write_parquet(out)
    log.info(f"Saved blended scores: {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["val","test"], required=True)
    args = ap.parse_args()
    main(args.config, args.stage)
