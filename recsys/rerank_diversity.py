import argparse, polars as pl
from .config import load_config
from .utils.logging import setup_logger

def mmr_rerank(df: pl.DataFrame, topk: int, mmr_lambda: float, category_col: str, brand_col: str, max_per_brand: int, max_per_category: int):
    # простой MMR без эмбеддингов: штрафуем за повторы брендов/категорий
    selected = []
    brand_cnt = {}
    cat_cnt = {}

    pool = df.sort("blend_score", descending=True).to_dicts()
    for cand in pool:
        if len(selected) >= topk: break
        b = cand.get(brand_col)
        c = cand.get(category_col)
        penalty = 0.0
        if b is not None and brand_cnt.get(b,0) >= max_per_brand: penalty += 1.0
        if c is not None and cat_cnt.get(c,0) >= max_per_category: penalty += 1.0
        score = mmr_lambda*cand["blend_score"] - (1-mmr_lambda)*penalty
        selected.append((score, cand))
        # обновим счётчики когда уже доберём топ по итоговым score
    selected = sorted(selected, key=lambda x: x[0], reverse=True)[:topk]
    out = []
    for _, cand in selected:
        b = cand.get(brand_col); c = cand.get(category_col)
        if b is not None: brand_cnt[b] = brand_cnt.get(b,0)+1
        if c is not None: cat_cnt[c] = cat_cnt.get(c,0)+1
        out.append(cand)
    return pl.DataFrame(out)

def main(config_path: str, stage: str):
    log = setup_logger("diversity")
    cfg = load_config(config_path).raw
    col = cfg["columns"]; paths = cfg["paths"]
    topk = cfg["submission"]["topk"]
    mmr_lambda = cfg["diversity"]["mmr_lambda"]
    max_per_brand = cfg["diversity"]["max_per_brand"]
    max_per_cat = cfg["diversity"]["max_per_category"]

    blended = pl.read_parquet(f"{paths['out_dir_processed']}/blended_{stage}.parquet")
    items = pl.scan_parquet(paths["items"]).select([col["item_id"], col["brand"], col["category_id"]]).collect()

    df = blended.join(items, on=col["item_id"], how="left")

    # применим по каждому пользователю
    reranked = []
    for uid, g in df.group_by(col["user_id"], maintain_order=True):
        rr = mmr_rerank(g, topk, mmr_lambda, col["category_id"], col["brand"], max_per_brand, max_per_cat)
        reranked.append(rr)
    out = pl.concat(reranked) if reranked else pl.DataFrame()
    out_path = f"{paths['out_dir_processed']}/final_{stage}.parquet"
    out.write_parquet(out_path)
    log.info(f"Saved reranked list: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["val","test"], required=True)
    args = ap.parse_args()
    main(args.config, args.stage)
