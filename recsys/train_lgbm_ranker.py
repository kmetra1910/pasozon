import argparse, lightgbm as lgb, pandas as pd, polars as pl, numpy as np, os
from sklearn.metrics import ndcg_score
from .config import load_config
from .utils.logging import setup_logger

def main(config_path: str):
    log = setup_logger("train_lgbm")
    cfg = load_config(config_path).raw
    col = cfg["columns"]; paths = cfg["paths"]

    feats_path = f"{paths['out_dir_processed']}/features_val.parquet"
    df = pl.read_parquet(feats_path).to_pandas()

    label = df.pop("label").values
    qid = df[col["user_id"]].values
    # простые фичи (выкинем id и string)
    drop = [col["user_id"], col["item_id"], "source"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")

    # группировка по пользователю
    _, group_sizes = np.unique(qid, return_counts=True)

    params = cfg["ranker"]["params"]
    train = lgb.Dataset(X, label=label, group=group_sizes, free_raw_data=False)
    model = lgb.train(params, train_set=train, num_boost_round=800)

    # offline NDCG@100
    y_pred = model.predict(X)
    # считаем по группам
    ndcgs = []
    start = 0
    for g in group_sizes:
        s, e = start, start+g
        ndcgs.append(ndcg_score([label[s:e]], [y_pred[s:e]], k=100))
        start = e
    log.info(f"Mean NDCG@100 (val): {np.mean(ndcgs):.5f}")

    os.makedirs(paths["models_dir"], exist_ok=True)
    model_path = f"{paths['models_dir']}/lgbm_ranker.txt"
    model.save_model(model_path)
    log.info(f"Saved model: {model_path}")

    # сохраним скор для blend
    out_scores = df[[col["user_id"], col["item_id"]]].copy()
    out_scores["lgbm_score"] = y_pred
    pl.from_pandas(out_scores).write_parquet(f"{paths['out_dir_processed']}/scores_val.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
