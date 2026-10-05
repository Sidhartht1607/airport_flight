# ✈️ Flight Delay Analytics & Prediction (Q1 2025)

## The problem

I wanted to know two things: *why* do flights get delayed, and *can I predict it before it happens?*

I pulled U.S. DOT flight data for January–March 2025 (`flight_jan.csv`, `flight_feb.csv`, `flight_mar.csv` — about 1.65M flights combined) and used it to dig into which airports, carriers, and times of day carry the most disruption risk, what actually causes the delays that do happen, and finally to train a model that predicts whether a given flight will depart 15+ minutes late (`dep_del15`) using only information known ahead of departure.

Everything below is a summary — the full reasoning, every intermediate check, and all the dead ends live in `flight_data_set.ipynb` as markdown notes right next to the code that produced them.

### Experiment log, at a glance

| # | Attempt | Reasoning | Result |
|---|---|---|---|
| 1 | Majority-class + logistic regression baseline | Needed to know what "doing nothing clever" scores before trusting any model number | 81.0% acc / 0.647 AUC — the floor everything else has to beat |
| 2 | PyTorch MLP, one-hot features | Wanted the practice writing a training loop by hand, and a template I could extend later | 81.5% acc / 0.714 AUC — beat the linear baseline on AUC, barely on accuracy |
| 3 | Same MLP, class-weighted loss | Assumed the weak accuracy gain was purely a class-imbalance problem | AUC unchanged (0.714) — recalibrated the threshold, not the ranking. Accuracy dropped to 66.1% |
| 4 | MLP with learned embeddings instead of one-hot, more epochs, early stopping | Class weighting didn't work, so gave the model new information to rank on instead | **82.1% acc / 0.730 AUC — the actual improvement** |

Each attempt is written up in full below and in the notebook, including why the failed one failed.

## What I found in the exploration

- Cancellations cluster at the start of the year — I'm guessing post-holiday weather and traffic, though I haven't confirmed that against a weather dataset.
- Delay risk isn't evenly spread across the day: mornings (05:00–09:00) are consistently low-risk (about 10% on average across the airport-hour buckets I trust, against 19.0% overall), delays build through the afternoon (about 23% for 15:00–19:00), and a few airports run much higher late in the evening: DCA at 22:00 (about 46%, 1,642 flights), BWI at 22:00 (about 42%) and PBI at 19:00 (about 42%). DFW stays at 26–32% from 20:00 to 22:59, well above average but not extreme. I only trusted this pattern after computing 95% confidence intervals per airport-hour bucket and dropping anything with fewer than 100 flights or a CI half-width over 7 points — the raw numbers looked noisier than they actually were. By that rule PIA, SFB and SGF drop out of the late-evening list (2 to 73 flights per evening hour), and no trusted bucket reaches 50% at any hour of the day (the highest is 49%).
- Of all recorded delay minutes, 77% trace back to *internal* causes (late aircraft turnaround, carrier issues) rather than weather or NAS/security delays — so most of the delay problem here is operational, not external.
- Some carrier/airport pairs run well above that carrier's own network-wide delay rate, which points at localized scheduling or capacity problems rather than the carrier being bad everywhere.

## The prediction problem, and why a baseline mattered

Before trusting any model number, I needed to know what "doing nothing clever" gets me. About 4 in 5 flights in this data are on time, so a model that always predicts "on time" already scores ~81% accuracy without knowing anything about the flight. That's the number every real model has to clear for accuracy to mean anything.

I ended up tracking two baselines and one real model, all logged to MLflow so I could compare them properly instead of trusting memory:

| Model | Test accuracy | Test AUC |
|---|---|---|
| Majority-class ("always predict on-time") | 81.03% | — |
| Logistic regression | 81.04% | 0.647 |
| **PyTorch MLP (2 hidden layers, 5 epochs)** | **81.55%** | **0.712** |

The honest takeaway: on accuracy alone, the neural net barely beats guessing, because ~80% accuracy is what you get for free from the class imbalance. AUC is the metric that actually shows something happened — logistic regression's 0.647 vs. the MLP's 0.712 tells me the extra model capacity is picking up on non-linear interactions (hour × origin, carrier × origin, etc.) that a linear model can't represent, which lines up with the heatmap patterns found during the EDA. It's a real, if modest, improvement — not a finished product.

## Does class weighting fix it?

My first guess was that the low accuracy-over-baseline was purely a class-imbalance problem, so I tried weighting the loss — `pos_weight ≈ 4.27` in `BCEWithLogitsLoss` (the ratio of on-time to delayed flights in the training set) — to make the model pay more for getting delayed flights wrong.

| Model | Test accuracy | Test AUC |
|---|---|---|
| PyTorch MLP (unweighted) | 81.50% | 0.714 |
| **PyTorch MLP (class-weighted)** | **66.10%** | **0.714** |

AUC didn't move (0.7138 → 0.7143 — noise), and accuracy collapsed from 81.5% to 66.1%. In hindsight this is exactly what should happen: AUC measures ranking quality across every threshold, and `pos_weight` doesn't teach the model anything new about which flights are riskier — it just shifts the model's output scores so more flights cross the fixed 0.5 cutoff I'm using in `evaluate()`. That's a threshold/calibration change, not a ranking one, so AUC (threshold-independent by definition) stayed flat while accuracy at that one fixed threshold cratered from the flood of new false positives.

So: **class weighting alone doesn't improve AUC.** If AUC is the target, the model needs new information to rank on — embeddings for the categorical columns, more epochs, or the engineered root-cause features are the more promising levers. Class weighting *would* be the right tool if the actual goal shifts to "catch more real delays and accept more false alarms" — but that requires picking a new decision threshold for the weighted model too, not reusing 0.5 from a differently-calibrated model.

## Embeddings: the one that actually worked

Next lever from the list: replace the one-hot-encoded categorical columns (`op_carrier`, `origin`, `origin_state_nm`, `dest`, `dest_state_nm`, `dep_time_blk`, `arr_time_blk`) with learned embeddings. One-hot encoding had blown the feature count up to 831 mostly-empty columns and gave the model no way to know that, say, two airports behave similarly — every category was an unrelated indicator column. An embedding table gives each category a small dense vector the model tunes during training, so similar-behaving airports/carriers can end up with similar vectors, and it scales much better than one-hot as cardinality grows (`origin`/`dest` have 334 categories each here).

I kept the same 80/20 train/test split as every prior model (`random_state=42`) for a fair comparison, and added a validation split carved out of training data purely for early stopping — the test set stays untouched until training is fully done, so the "when to stop" decision doesn't leak test information into the reported numbers.

| Model | Test accuracy | Test AUC |
|---|---|---|
| PyTorch MLP (one-hot, unweighted) | 81.50% | 0.714 |
| PyTorch MLP (one-hot, class-weighted) | 66.10% | 0.714 |
| **PyTorch MLP + embeddings (early-stopped, 45 epochs)** | **82.08%** | **0.730** |

This time both metrics moved together — +1.6 points of AUC *and* better accuracy, not a threshold trade-off like class weighting was. It also needed a completely different training regime to get there: 5 epochs (enough for the one-hot model) barely gave the embeddings time to organize; this version trained up to 45 epochs with dropout + batch norm added to control overfitting, and used validation-AUC early stopping instead of a fixed, guessed epoch count.

Logged as the `pytorch_embedding_mlp` run in the same `flight-departure-delay` MLflow experiment, so it's directly comparable to every run before it.

## Why PyTorch and why MLflow

I could have gotten similar or slightly better numbers out of `sklearn`'s built-in classifiers with a fraction of the code. I chose PyTorch on purpose:

- I wanted the practice writing an actual training loop by hand instead of calling `.fit()`.
- It gives me a template I can extend later — e.g., replacing the one-hot-encoded airport/carrier columns (which currently blow the feature count up to 831 mostly-sparse columns) with learned embeddings, which isn't really an option inside `sklearn`.

MLflow came in because I kept re-running the notebook while tweaking hyperparameters and losing track of which run produced which numbers. Now every run — params, per-epoch metrics, and the trained model artifact — gets logged automatically under the `flight-departure-delay` experiment, so comparing runs is a UI click instead of scrollback archaeology.

To view the tracked experiments locally:

```bash
source myenv/bin/activate
mlflow ui --backend-store-uri mlruns
```

(This MLflow version defaults to requiring a database backend; if you hit a "filesystem tracking backend is in maintenance mode" error, either run `mlflow ui --backend-store-uri sqlite:///mlflow.db` after migrating with `mlflow migrate-filestore`, or set `MLFLOW_ALLOW_FILE_STORE=true` to keep using the plain `mlruns/` folder as-is.)

## Problems I ran into (and how I fixed them)

- **`op_carrier` didn't exist.** A bunch of cells referenced `df.op_carrier`, but the raw column from DOT is `OP_UNIQUE_CARRIER`, and my rename step only ever renamed `DAY_OF_MONTH`. Every carrier-related cell would have thrown a `KeyError` if actually run top to bottom. Fixed by adding `OP_UNIQUE_CARRIER` to the same rename step.
- **Sparse matrix into PyTorch.** `ColumnTransformer.fit_transform` returns a sparse matrix once a `OneHotEncoder` is involved, and `torch.from_numpy` doesn't accept scipy sparse matrices — first run threw a `TypeError`. Fixed with a `.toarray()` call right after the transform. Not the most memory-efficient choice at ~1.3M rows, but it fits fine in RAM at this dataset size, so I didn't over-engineer a sparse-tensor path.
- **`mlflow.pytorch.log_model` failed on the first save**, with `MlflowException: If serialization_format is set to 'pt2', then input_example is required...`. This MLflow version defaults PyTorch model logging to the `pt2` traced-graph format, which needs a sample input to trace the forward pass. Fixed by passing `input_example=X_train[:5]` into `log_model`.
- **`mlflow.pytorch.log_model` failed again for the embedding model**, this time with `Unsupported signature type for the selected serialization format... 'pt2'`. The embedding model's `forward` takes *two* tensors (numeric + categorical indices), and the `pt2` traced-graph tracer can't handle a multi-input signature the way it handled the single-tensor one-hot model. Fixed by passing `serialization_format="pickle"` instead — MLflow warns that pickle formats can execute arbitrary code on load, which is a fair caution for a shared/public registry, but it's fine for a model I trained and am loading myself.
- **Cancelled flights polluting delay stats.** Cancelled/diverted flights don't have a meaningful `dep_del15`, so I excluded them from `df_operation` before any delay-rate or modelling work — otherwise every downstream rate calculation would have been skewed by rows that never really departed.
- **Small-sample airports/hours looked scarier than they were.** Some airport-hour buckets had a handful of flights and wildly high "delay rates" purely from small-sample noise. I only trusted a bucket once it had at least 100 flights and a 95% CI half-width under 7 points.

## Packages used, and why

| Package | What it's for here |
|---|---|
| `pandas`, `numpy` | Loading, joining, and cleaning the three monthly CSVs; all the groupby/aggregation work in the EDA. |
| `matplotlib`, `seaborn` | Donut charts, bar charts, and the airport × hour delay-rate heatmaps. |
| `scikit-learn` | Preprocessing plumbing for the model (`train_test_split`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`) and the logistic regression baseline — not used for the actual prediction model. |
| `torch` | The actual prediction model — first a small 2-hidden-layer MLP on one-hot features, later a version with learned embeddings for the categorical columns instead. Chosen over sklearn for the practice writing the training loop by hand, and because embeddings aren't really an option inside sklearn. |
| `mlflow` | Experiment tracking — params, per-epoch metrics, and the model artifact for every run, so runs are comparable without manual note-taking. |

## What's next

1. **Feed in the root-cause features from the EDA** (delay-by-hour, airport delay-cause shares) as engineered inputs — that part of the notebook already found real, non-random structure the model still isn't given directly.
2. **Wider embeddings for `origin`/`dest` specifically** — those two 334-category columns carry most of the geographic signal but are currently capped by the same sizing formula as much smaller columns like `op_carrier` (15 categories).
3. **A learning-rate schedule** (e.g. reduce-on-plateau tied to validation AUC) — training loss was still inching down at epoch 45 without early stopping ever triggering, suggesting there's a bit more headroom than flat-rate epochs alone are capturing.
4. **Revisit class weighting on top of the embedding model** if the goal shifts from "maximize AUC" to "catch more real delays" — now that ranking quality has genuinely improved, weighting is worth retrying, this time paired with a properly chosen threshold instead of reusing 0.5.

## Getting the data

The raw CSVs (`flight_jan.csv`, `flight_feb.csv`, `flight_mar.csv`, ~137–163MB each) aren't in this repo — well over GitHub's 100MB per-file limit, and not worth a Git LFS setup for a dataset this replaceable. Pull them yourself from the DOT Bureau of Transportation Statistics "Reporting Carrier On-Time Performance" download tool (the exact query used is linked at the top of `requirements.txt`): pick On-Time Performance data for January, February, and March 2025, download each as CSV, and drop them in this folder with those filenames before running the notebook.

## Repo layout

```
airport_flight/
├── flight_data_set.ipynb     # main notebook — EDA, root-cause analysis, and the PyTorch/MLflow model
├── flight_data_set12.ipynb   # earlier working copy
├── flight_jan.csv / flight_feb.csv / flight_mar.csv   # raw DOT data, Q1 2025 (gitignored — see "Getting the data")
├── requirements.txt
├── myenv/                    # local virtualenv (gitignored)
└── mlruns/                   # MLflow tracking data — committed, since it's small and is the whole point of the experiment log above
```
