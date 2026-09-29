# What changed

- Original notebook retained.
- Production feature selection is now: 34-feature LightGBM LambdaRank -> LightGBM gain importance -> top 15 -> final 15-feature LambdaRank.
- Production preserves the notebook live cascade: A/B/C/E/F retrieval, production-only validity filtering, 80/20 exploit/explore, and live session updates.
- Existing Gradio visual code/CSS was preserved. Only login/email element IDs and a small JS API bridge were added so the existing "For you" cards become dynamic and clickable.
- FastAPI adds recommendation and click endpoints.
- SQLite stores users/offers/click events.
- Redis stores live session state and short recommendation cache.
