from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


class Recommender:
    def __init__(self, bundle_path: str | os.PathLike):
        self.b = joblib.load(bundle_path)
        self.model = self.b["model"]
        self.item_ids = np.asarray(self.b["item_ids"])
        self.n_items = len(self.item_ids)
        self.offer_features = self.b["offer_features"]
        self.user_profiles = self.b["user_profiles"].set_index("user_id")
        self.user_id_to_row = self.b["user_id_to_row"]
        self.item_index = self.b["item_index"]
        self.last_live_day = int(self.b["global_eval_day"])
        self.anchor_date = pd.Timestamp(self.b["anchor_date"])

    # ---------- lookup / session defaults ---------------------------------
    def known_user(self, user_id: str) -> bool:
        return user_id in self.user_id_to_row

    def _user_row(self, user_id: str):
        return self.user_id_to_row.get(user_id)

    def initial_session(self, user_id: str) -> dict:
        macro_n = len(self.b["macro_cats"])
        partner_n = self.b["n_partners"] + 1
        gt_n = len(self.b["gt_idx"])
        row = self._user_row(user_id)
        if row is not None:
            uf = self.user_profiles.loc[user_id]
            cat = np.asarray(uf[self.b["cat_pref_cols"]], dtype=np.float32)
            partner = np.asarray(uf[self.b["partner_pref_cols"]], dtype=np.float32)
            gt = np.asarray(uf[self.b["gt_pref_cols"]], dtype=np.float32)
            if len(partner) < partner_n:
                partner = np.pad(partner, (0, partner_n - len(partner)))
            open_intensity = float(self.b["raw_open_per_user"].get(user_id, 0.0))
            purch_intensity = float(self.b["raw_purch_per_user"].get(user_id, 0.0))
        else:
            cat = np.zeros(macro_n, dtype=np.float32)
            partner = np.zeros(partner_n, dtype=np.float32)
            gt = np.zeros(gt_n, dtype=np.float32)
            open_intensity = 0.0
            purch_intensity = 0.0

        return {
            "live_cat_pref": cat,
            "live_partner_pref": partner,
            "live_gt_pref": gt,
            "active_session_clicks": [],
            "tfidf_profile": np.zeros(self.b["offer_tfidf_dense"].shape[1], dtype=np.float32),
            "tfidf_wsum": 0.0,
            "open_intensity": open_intensity,
            "purch_intensity": purch_intensity,
            "current_live_day": self.current_live_day(),
        }

    def on_click(self, state: dict, offer_id: str, weight: float = 1.0):
        if offer_id not in self.item_index:
            raise KeyError(f"Unknown offer_id: {offer_id}")
        i = int(self.item_index[offer_id])
        row = self.offer_features.iloc[i]
        cat_i = self.b["macro_idx"].get(row["cat_san"])
        partner_i = self.b["partner_idx"].get(row["partner_san"], self.b["n_partners"])
        gt_i = self.b["gt_idx"].get(row["gt_san"])
        if cat_i is not None:
            state["live_cat_pref"][cat_i] += weight
        state["live_partner_pref"][partner_i] += weight
        if gt_i is not None:
            state["live_gt_pref"][gt_i] += weight
        state["active_session_clicks"].append(offer_id)
        state["tfidf_profile"] += weight * self.b["offer_tfidf_dense"][i]
        state["tfidf_wsum"] += float(weight)
        # Same live update convention used by the notebook for an open event.
        state["open_intensity"] = max(state["open_intensity"] - 1.0, 0.0)
        return state

    # ---------- retrieval --------------------------------------------------
    def _topk(self, scores, k):
        k = min(int(k), self.n_items)
        if k <= 0:
            return np.array([], dtype=int)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]

    def channel_a(self, seeds, top_k=120):
        if not seeds:
            return np.array([], dtype=int)
        sim = self.b["item_item_sim"]
        agg = np.asarray(sim[seeds].sum(axis=0)).ravel()
        agg[np.asarray(seeds, dtype=int)] = -1.0
        positive = int((agg > 0).sum())
        if not positive:
            return np.array([], dtype=int)
        return self._topk(agg, min(top_k, positive))

    def channel_b(self, user_id, top_k=60):
        gov = self.b["user_gov_per_user"].get(user_id, "unknown")
        row = self._user_row(user_id)
        top_cat = self.b["macro_cats"][0]
        if row is not None:
            top_cat = self.b["macro_cats"][int(self.b["top_spend_cat_idx"][row])]
        gov_items = self.b["gov_top_items"].get(gov, np.array([], dtype=int))[:15]
        set_items = self.b["govset_top_items"].get(gov, np.array([], dtype=int))[:15]
        cat_items = self.b["cat_top_items"].get(top_cat, np.array([], dtype=int))[:15]
        nat_items = self.b["nationwide_top_items"][:40]
        merged = np.concatenate([gov_items, set_items, cat_items, nat_items])
        _, first = np.unique(merged, return_index=True)
        return merged[np.sort(first)][:top_k]

    def channel_c(self, user_id, state, top_k=40):
        row = self._user_row(user_id)
        base = self.b["content_sim_full"][row] if row is not None else np.zeros(self.n_items, dtype=np.float32)
        tp = state["tfidf_profile"]
        norm = np.linalg.norm(tp)
        if state["tfidf_wsum"] > 0 and norm > 0:
            txt = (self.b["offer_tfidf_dense"] @ tp) / norm
        else:
            txt = np.zeros(self.n_items, dtype=np.float32)
        scores = 0.6 * base + 0.4 * np.asarray(txt, dtype=np.float32)
        return self._topk(scores, top_k)

    def channel_e(self, seeds, top_partners=10, per_partner=3):
        if not seeds:
            return np.array([], dtype=int)
        pidx = self.b["offer_partner_idx"][np.asarray(seeds, dtype=int)]
        parts, counts = np.unique(pidx, return_counts=True)
        out = []
        for p in parts[np.argsort(-counts)[:top_partners]]:
            arr = self.b["partner_top_items"].get(int(p))
            if arr is None:
                continue
            arr = arr[~np.isin(arr, seeds)]
            if len(arr):
                out.append(arr[:per_partner])
        return np.unique(np.concatenate(out)) if out else np.array([], dtype=int)

    def channel_f(self, user_id, seeds, top_k=60):
        code = self.b["user_row_to_cell"].get(user_id)
        cnt = self.b["demo_cell_cnt"].get(code)
        if cnt is None or not np.any(cnt):
            return np.array([], dtype=int)
        cnt = cnt.copy()
        own = self.b["user_items_map"].get(user_id)
        if own is not None and len(own):
            cnt[own] -= 1.0
            cnt = np.maximum(cnt, 0.0)
        top = self._topk(cnt, min(top_k, int((cnt > 0).sum()))) if (cnt > 0).any() else np.array([], dtype=int)
        return top[~np.isin(top, seeds)]

    def candidates(self, user_id, state):
        seeds = [self.item_index[x] for x in state["active_session_clicks"] if x in self.item_index]
        a = self.channel_a(seeds)
        b = self.channel_b(user_id)
        c = self.channel_c(user_id, state)
        e = self.channel_e(seeds)
        f = self.channel_f(user_id, seeds)
        return np.unique(np.concatenate([a, b, c, e, f]))

    # ---------- 15-feature live table -------------------------------------
    def _geo_match(self, user_id, item_idx):
        gov = self.b["user_gov_per_user"].get(user_id, "unknown")
        gi = self.b["user_gov_tag_idx_map"].get(user_id, -1)
        prim = self.b["offer_gov_san_arr"][item_idx]
        out = np.where(prim == gov, 2.0, 0.0).astype(np.float32)
        if gi >= 0:
            inset = self.b["gov_set_membership"][item_idx, gi] > 0
            out = np.where((out == 0) & inset, 1.5, out)
            ni = self.b["nationwide_tag_idx"]
            if ni >= 0:
                nation = self.b["gov_set_membership"][item_idx, ni] > 0
                out = np.where((out == 0) & nation, 1.0, out)
        return out

    def live_features(self, user_id, item_idx, state):
        item_idx = np.asarray(item_idx, dtype=int)
        n = len(item_idx)
        row = self._user_row(user_id)
        cats = self.b["offer_cat_idx"][item_idx]
        partners = self.b["offer_partner_idx"][item_idx]
        cat_pref = state["live_cat_pref"]
        partner_pref = state["live_partner_pref"]
        gt_pref = state["live_gt_pref"]
        wa_pref = gt_pref[self.b["waff_gt_idx"]] if self.b["waff_gt_idx"] is not None else 0.0
        if row is not None:
            spend = self.b["user_spend_amt"][row, cats].astype(np.float32)
            affluence = float(self.b["user_affluence_score"][row])
            top_spend = int(self.b["top_spend_cat_idx"][row])
            flags = self.b["has_flags"][row]
            total_spend = float(self.b["total_merchant_spend_per_user"][row])
        else:
            spend = np.zeros(n, dtype=np.float32)
            affluence = 1.0
            top_spend = -1
            flags = np.zeros(4, dtype=np.float32)
            total_spend = 0.0

        tp = state["tfidf_profile"]
        tp_norm = np.linalg.norm(tp)
        if state["tfidf_wsum"] > 0 and tp_norm > 0:
            text_sim = (self.b["offer_tfidf_dense"][item_idx] @ tp) / tp_norm
            text_sim = np.clip(text_sim, 0.0, 1.0).astype(np.float32)
        else:
            text_sim = np.zeros(n, dtype=np.float32)

        day = float(state["current_live_day"])
        of = self.b
        df = pd.DataFrame({
            "log_price": of["offer_log_price"][item_idx],
            "discount_norm": of["offer_discount_norm"][item_idx],
            "signal_quality": of["offer_signal_quality"][item_idx],
            "log_popularity": of["offer_log_popularity"][item_idx],
            "offer_purchase_rate": of["offer_purchase_rate"][item_idx],
            "offer_redemption_rate": of["offer_redemption_rate"][item_idx],
            "cash_amount_norm": of["offer_cash_amount_norm"][item_idx],
            "dtype_percentage": of["offer_dtype_pct"][item_idx],
            "dtype_fixed": of["offer_dtype_fixed"][item_idx],
            "dtype_unknown": of["offer_dtype_unknown"][item_idx],
            "is_waffarha_offer": of["offer_is_waffarha"][item_idx],
            "expanded_geo_match": self._geo_match(user_id, item_idx),
            "discount_cash_impact": (of["offer_gift_price"][item_idx] * of["offer_discount_norm"][item_idx]).astype(np.float32),
            "log_days_to_expire": np.log1p(np.clip(of["gift_end_day_rel"][item_idx] - day, 0, None)).astype(np.float32),
            "yc_category_share": (spend / (total_spend + 1.0)).astype(np.float32),
            "description_tfidf_sim": text_sim,
            "user_30d_open_intensity": np.full(n, state["open_intensity"], dtype=np.float32),
            "user_30d_conversion_ratio": np.full(n, state["purch_intensity"] / (state["open_intensity"] + 1.0), dtype=np.float32),
            "cat_affinity_score": cat_pref[cats],
            "waffarha_synergy": wa_pref * of["offer_is_waffarha"][item_idx],
            "cat_match_top_spend": (cats == top_spend).astype(np.float32),
            "retrieval_channel_count": np.full(n, 2.0, dtype=np.float32),
            "user_candidate_partner_affinity": partner_pref[partners],
            "user_spend_in_candidate_category": spend,
            "affordability_ratio": (of["offer_log_price"][item_idx] / affluence).astype(np.float32),
            "user_has_yc": np.full(n, flags[0], dtype=np.float32),
            "user_has_bnpl": np.full(n, flags[1], dtype=np.float32),
            "user_has_credit": np.full(n, flags[2], dtype=np.float32),
            "user_has_debit": np.full(n, flags[3], dtype=np.float32),
            "user_total_interactions": np.full(n, cat_pref.sum() + gt_pref.sum(), dtype=np.float32),
            "user_avg_score": np.zeros(n, dtype=np.float32),
            "user_max_score": np.zeros(n, dtype=np.float32),
            "user_waffarha_affinity": np.full(n, wa_pref, dtype=np.float32),
            "user_redemption_rate_hist": np.zeros(n, dtype=np.float32),
        })
        return df[self.b["selected_features"]]

    def current_live_day(self):
        mode = os.getenv("LIVE_DAY", str(self.last_live_day)).strip().lower()
        if mode in ("dataset", "default"):
            return self.last_live_day
        if mode == "auto":
            return int((date.today() - self.anchor_date.date()).days)
        return int(mode)

    def recommend(self, user_id, state, top_n=10, explore_frac=0.2):
        cands = self.candidates(user_id, state)
        if len(cands) == 0:
            cands = np.argsort(-self.b["offer_log_popularity"])
        # Production validity filter sits immediately before reranking.
        cands = self._active_filter_with_day(cands, self.current_live_day(), top_n)
        if len(cands) == 0:
            return []
        X = self.live_features(user_id, cands, state)
        scores = np.asarray(self.model.predict(X))
        order = np.argsort(-scores)
        n_exploit = max(1, int(round(top_n * (1 - explore_frac))))
        exploit = cands[order[:n_exploit]]

        # Small deterministic exploration tail: high-quality valid offers from
        # categories not already represented in exploitation.
        exploit_cats = set(self.b["offer_cat_idx"][exploit].tolist())
        gov = self.b["user_gov_per_user"].get(user_id, "unknown")
        day = self.current_live_day()
        mask = (self.b["offer_gov_san_arr"] == gov) | (self.b["offer_gov_san_arr"] == "nationwide")
        mask &= (self.b["gift_start_day_rel"] <= day) & (day <= self.b["gift_end_day_rel"])
        idx = np.where(mask & ~np.isin(np.arange(self.n_items), exploit))[0]
        idx = idx[~np.isin(self.b["offer_cat_idx"][idx], list(exploit_cats))]
        n_explore = max(0, top_n - n_exploit)
        if n_explore and len(idx):
            quality = self.b["offer_signal_quality"][idx] * 10 + self.b["offer_log_popularity"][idx]
            pick = self._topk(quality, min(n_explore, len(idx)))
            final = np.concatenate([exploit, idx[pick]])[:top_n]
        else:
            final = exploit[:top_n]

        out = []
        for i in final:
            r = self.offer_features.iloc[int(i)]
            desc = str(r.get("gift_description", "")).strip()
            partner = str(r.get("offer_partner_en", r.get("partner_san", "Offer"))).strip()
            title = partner or str(r.get("offer_id", "Offer"))
            subtitle = desc[:90] if desc else str(r.get("cat_san", "Special offer"))
            out.append({"offer_id": str(r["offer_id"]), "title": title, "subtitle": subtitle, "score": float(scores[np.where(cands == i)[0][0]]) if i in cands else 0.0})
        return out

    def _active_filter_with_day(self, candidates, day, top_n):
        candidates = np.asarray(candidates, dtype=int)
        active = (self.b["gift_start_day_rel"][candidates] <= day) & (day <= self.b["gift_end_day_rel"][candidates])
        kept = candidates[active]
        if len(kept) >= top_n:
            return kept
        rest = candidates[~active]
        if len(rest):
            rest = rest[np.argsort(-self.b["offer_log_popularity"][rest])]
            kept = np.concatenate([kept, rest])
        return kept[:top_n]
