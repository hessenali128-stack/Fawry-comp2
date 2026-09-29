# Fawry Recommendation App — FastAPI + Gradio + Redis + SQLite

This project keeps the notebook's recommendation logic and the existing Gradio visual design, with one model change:

1. Train the original 34-feature LambdaRank selector.
2. Select the top 15 features using LightGBM **gain importance**.
3. Retrain the final LambdaRank using only those 15 features.
4. Save the model + preprocessing/retrieval artifacts into one `model_bundle.joblib`.
5. Production runs FastAPI + the unchanged Gradio design in one app container; SQLite is local to that container/volume and Redis is the cache service.

## Folder layout

```text
notebook/       original notebook
training/       one-off training container
production/     FastAPI + recommender + production container
ui/             original Gradio UI, only minimally wired to the API
artifacts/      model bundle after training
data/           SQLite database
```

## GitHub Codespaces — train

Put these three CSVs under `dataset/`:

```text
user_features.csv
train_interactions.csv
offer_features.csv
```

Run:

```bash
docker compose --profile train run --rm trainer
```

The trainer writes:

```text
artifacts/model_bundle.joblib
artifacts/selected_features.json
```

The selected 15 are printed in the terminal.

## GitHub Codespaces — production

```bash
docker compose up -d --build app redis
```

Open port **7860** in Codespaces.

The same container serves:

```text
GET  /api/health
GET  /api/features
GET  /api/recommendations/{user_id}
POST /api/offers/click
GET  /             -> Gradio UI
```

## Click -> instant re-rank

The current Gradio cards are visually unchanged. A tiny JS bridge sends:

```json
{"user_id":"...","offer_id":"..."}
```

to `POST /api/offers/click`.

The backend then:

```text
click
  -> update live category/partner/gift-type + TF-IDF session state
  -> save session in Redis
  -> invalidate cached recommendations
  -> run candidate retrieval
  -> apply production-only offer validity
  -> run the 15-feature LightGBM ranker
  -> return new Top-10
```

That is the same live flywheel used in the notebook, without changing the Gradio styling.

## Notes

- `LIVE_DAY=dataset` keeps the same relative evaluation day used by the notebook.
- Set `LIVE_DAY=auto` when the dataset dates correspond to real calendar dates.
- For a dataset user, logging in with the actual `user_id` (for example `CUST_1168`) gives the static profile. Unknown IDs still work as cold-start users.
- The production app container contains FastAPI + Gradio + model + SQLite. Redis is kept as a separate service, which is the normal container layout for a Redis daemon.
