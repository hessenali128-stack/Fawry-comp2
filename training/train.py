import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = Path(os.getenv("NOTEBOOK", ROOT / "notebook" / "Fawry_Recommendation_System_educate.ipynb"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "dataset"))
OUT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The notebook uses a relative dataset path. Keep execution isolated in the
# project root so the original cells remain unchanged.
os.chdir(ROOT)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}. Put the notebook CSV files there.")

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
code_cells = ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]
# Execute only through the complete 34-feature table (cell 119 in the notebook).
# The notebook's v4 ablation and benchmark/sandbox are intentionally skipped for
# production training to keep the trainer small and deterministic.
code = []
for cell_no, src in enumerate(code_cells):
    code.append(src)
    if "FEATURE_COLS_LEGACY =" in src:
        break

ns = {"__file__": str(NOTEBOOK), "__name__": "__fawry_notebook__"}
exec("\n\n".join(code), ns)

pairwise = ns["pairwise_features"]
feature_cols = list(ns["FEATURE_COLS"])
train_fn = ns["train_lambdarank"]

print(f"Training baseline ranker on {len(feature_cols)} features...")
base_ranker, _ = train_fn(pairwise, feature_cols, "34-feature selector")

gain = base_ranker.booster_.feature_importance(importance_type="gain")
importance = pd.Series(gain, index=feature_cols).sort_values(ascending=False)
top15 = importance.head(15).index.tolist()

print("\nSelected top 15 features by LightGBM gain importance:")
for i, name in enumerate(top15, 1):
    print(f"{i:02d}. {name}: {importance[name]:.6f}")

print("\nTraining final production LambdaRank on top 15...")
final_ranker, _ = train_fn(pairwise, top15, "15-feature production")

# --- Build a slim, self-contained inference bundle -------------------------
# Keep only the data actually needed by the real-time path.
user_features = ns["user_features"]
cat_pref_cols = list(ns["cat_pref_cols"])
partner_pref_cols = list(ns["partner_pref_cols"])
gt_pref_cols = list(ns["gt_pref_cols"])
spend_amt_cols = list(ns["spend_amt_cols"])
user_cols = [
    "user_id", "governorate", "age_bucket", "gender",
    *cat_pref_cols, *partner_pref_cols, *gt_pref_cols, *spend_amt_cols,
    "uv_has_yc", "uv_has_bnpl", "uv_has_credit", "uv_has_debit",
]
user_cols = [c for c in user_cols if c in user_features.columns]
user_profiles = user_features[user_cols].copy()

# Dicts with numpy arrays are cheap to serialize and much faster than rebuilding
# them from the CSV files on every API startup.
user_items_map = {
    str(u): arr for u, arr in ns["_user_items_map"].items()
}

bundle = {
    "model": final_ranker,
    "selected_features": top15,
    "selected_feature_importance": {k: float(importance[k]) for k in top15},
    "all_feature_importance": {k: float(importance[k]) for k in feature_cols},
    "item_ids": ns["ITEM_IDS"],
    "offer_features": ns["offer_features"],
    "user_profiles": user_profiles,
    "cat_pref_cols": cat_pref_cols,
    "partner_pref_cols": partner_pref_cols,
    "gt_pref_cols": gt_pref_cols,
    "spend_amt_cols": spend_amt_cols,
    "anchor_date": str(pd.Timestamp(ns["ANCHOR_DATE"]).date()),
    "global_eval_day": int(ns["GLOBAL_EVAL_DAY"]),
    "macro_cats": ns["MACRO_CATS"],
    "macro_idx": ns["MACRO_IDX"],
    "partner_idx": ns["PARTNER_IDX"],
    "n_partners": int(ns["N_PARTNERS"]),
    "gt_idx": ns["GT_IDX"],
    "waff_gt_idx": ns["WAFF_GT_IDX"],
    "item_index": ns["ITEM_INDEX"],
    "user_id_to_row": ns["USER_ID_TO_ROW"],
    "user_spend_amt": ns["user_spend_amt"],
    "user_spend_norm": ns["user_spend_norm"],
    "user_affluence_score": ns["user_affluence_score"],
    "total_merchant_spend_per_user": ns["total_merchant_spend_per_user"],
    "top_spend_cat_idx": ns["top_spend_cat_idx"],
    "top_spend_cat_per_user": ns["top_spend_cat_per_user"],
    "user_gov_per_user": ns["user_gov_per_user"],
    "user_gov_tag_idx_map": ns["user_gov_tag_idx_map"],
    "has_flags": ns["has_flags"],
    "raw_open_per_user": dict(ns["raw_open_per_user"]),
    "raw_purch_per_user": dict(ns["raw_purch_per_user"]),
    "offer_cat_idx": ns["offer_cat_idx"],
    "offer_partner_idx": ns["offer_partner_idx"],
    "offer_is_waffarha": ns["offer_is_waffarha"],
    "offer_signal_quality": ns["offer_signal_quality"],
    "offer_discount_norm": ns["offer_discount_norm"],
    "offer_purchase_rate": ns["offer_purchase_rate"],
    "offer_redemption_rate": ns["offer_redemption_rate"],
    "offer_cash_amount_norm": ns["offer_cash_amount_norm"],
    "offer_log_price": ns["offer_log_price"],
    "offer_log_popularity": ns["offer_log_popularity"],
    "offer_dtype_pct": ns["offer_dtype_pct"],
    "offer_dtype_fixed": ns["offer_dtype_fixed"],
    "offer_dtype_unknown": ns["offer_dtype_unknown"],
    "offer_gift_price": ns["offer_gift_price"],
    "offer_gov_san_arr": ns["offer_gov_san_arr"],
    "gift_start_day_rel": ns["gift_start_day_rel"],
    "gift_end_day_rel": ns["gift_end_day_rel"],
    "gov_tag_idx": ns["GOV_TAG_IDX"],
    "nationwide_tag_idx": int(ns["NATIONWIDE_TAG_IDX"]),
    "gov_set_membership": ns["gov_set_membership"],
    "offer_tfidf_dense": ns["offer_tfidf_dense"],
    "content_sim_full": ns["content_sim_full"],
    "item_item_sim": ns["item_item_sim"],
    "gov_top_items": ns["gov_top_items"],
    "nationwide_top_items": ns["nationwide_top_items"],
    "cat_top_items": ns["cat_top_items"],
    "govset_top_items": ns["govset_top_items"],
    "partner_top_items": ns["partner_top_items"],
    "demo_cell_cnt": ns["DEMO_CELL_CNT"],
    "user_row_to_cell": ns["USER_ROW_TO_CELL"],
    "user_items_map": user_items_map,
}

path = OUT_DIR / "model_bundle.joblib"
joblib.dump(bundle, path, compress=3)
(OUT_DIR / "selected_features.json").write_text(json.dumps({
    "selected_features": top15,
    "importance_type": "gain",
    "importance": {k: float(importance[k]) for k in top15},
}, indent=2), encoding="utf-8")
print(f"\nSaved production bundle: {path}")
print(f"Bundle size: {path.stat().st_size / 1024 / 1024:.1f} MB")
