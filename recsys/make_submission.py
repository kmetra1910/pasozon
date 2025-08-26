import argparse, polars as pl
from .config import load_config
from .utils.logging import setup_logger

def main(config_path: str, stage: str):
    log = setup_logger("submission")
    cfg = load_config(config_path).raw
    col = cfg["columns"]; paths = cfg["paths"]; topk = cfg["submission"]["topk"]

    df = pl.read_parquet(f"{paths['out_dir_processed']}/final_{stage}.parquet")
    # гарантируем topk и формат
    df = (
        df.sort(["user_id","blend_score"], descending=[False, True])
          .group_by(col["user_id"]).head(topk)
          .group_by(col["user_id"])
          .agg(pl.col(col["item_id"]).alias("items"))
    )

    # собираем строку item_id_1 ... item_id_100 через пробел
    def join_items(xs): return " ".join(map(str, xs))
    sub = df.with_columns(
        pl.col("items").map_elements(join_items).alias("items_str")
    ).select([col["user_id"], "items_str"])

    out_path = cfg["submission"][f"filename_{stage}"]
    sub = sub.rename({col["user_id"]:"user_id", "items_str":"item_ids"})
    sub.write_csv(out_path)
    log.info(f"Saved submission: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["val","test"], required=True)
    args = ap.parse_args()
    main(args.config, args.stage) 
