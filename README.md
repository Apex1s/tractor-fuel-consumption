# Tractor Fuel Consumption Prediction

Predicting diesel consumption per agricultural intervention from GPS traces,
parcel geometry and equipment specifications, so that abnormal consumption —
mechanical faults, or fuel siphoned off between jobs — can be flagged against
what a job of that size *should* have taken.

**Result: 0.57 L mean absolute error on parcels the model never saw, against a
3.91 L naive baseline — a 6.9× improvement.** Cross-validated MAE is
0.47 ± 0.22 L on a target whose median is 1.09 L.

| Model | MAE (L) | vs. baseline |
|---|---|---|
| Predict the median | 3.056 ± 0.922 | 1.0× |
| Linear regression, duration only | 0.884 ± 0.322 | 3.5× |
| Linear regression, all features | 0.709 ± 0.121 | 4.3× |
| Gradient boosting, hand-tuned | 0.615 ± 0.210 | 5.0× |
| **Gradient boosting, defaults** | **0.472 ± 0.220** | **6.5×** |

*5-fold cross-validation on the training split, grouped by parcel: every
intervention on a parcel falls in the same fold, and the holdout is six whole
parcels (85 interventions). Jobs on one parcel share terrain and distances, so a
random split would score the model on near-copies of its training rows and
overstate accuracy. The hand-tuned configuration reaches a training MAE of
0.0002 — it memorises the data — and generalises worse than untouched defaults,
so the defaults are what ships.*

![Model comparison](reports/figures/model_comparison.png)

## The problem, stated precisely

Duration and distance are recorded *while a job runs*. That makes them valid
inputs for **retrospective anomaly detection** and invalid for **forward fuel
planning**, where neither is known in advance. This project targets the first.
Using it for the second would require a different feature set, and the
distinction is enforced throughout rather than glossed over.

## Data

500 labelled interventions across 30 parcels, from three sources:

| Source | Contents |
|---|---|
| `interventions_train.csv` | Machine, engine power, tool, tool width, parcel, operation type, fuel consumed |
| `trajets_train/` | One GPS trace per intervention: timestamp, latitude, longitude, speed |
| `Parcelles/` | Boundary vertices of each parcel |

Twelve features are engineered from these: duration, geodesic distance, three
speed summaries, two acceleration summaries, and parcel area, perimeter and
vertex count (areas computed after reprojecting each polygon to its local UTM
zone, since degrees are not a unit of area).

> **Note on completeness.** `data/raw/trajets_train/` holds a 100-file sample of
> the 500 training traces; the full set is not redistributed. The complete
> engineered feature table is committed at `data/processed/merged_data.csv`, and
> notebook 1 verifies that the pipeline reproduces it to floating-point
> precision on the 100 overlapping rows. All published results are reproducible
> from this repository. One parcel outline is genuinely absent from the source
> data, affecting 4 interventions, which are median-imputed.

`interventions_test.csv` has no target column — it is an unlabelled holdout, so
it yields predictions, not scores. Traces are shipped for 100 of its 320
interventions; `predictions.csv` lists all 320, with a prediction for the 100
that have a trace and `has_trajectory = False` for the others.

## Running it

```bash
git clone https://github.com/AnasSABBAHI/tractor-fuel-consumption
cd tractor-fuel-consumption

python -m venv venv              # Python 3.12, see .python-version
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter lab
pytest tests                    # after pip install pytest
```

Run the notebooks in order:

1. `notebooks/1_feature_engineering_and_eda.ipynb` — builds features from raw
   traces, validates them against the stored table, explores the data
2. `notebooks/2_modeling.ipynb` — baselines, cross-validation, final holdout
   evaluation, predictions on the unlabelled test set

No paths need editing. Figures are written to `reports/figures/`, predictions to
`data/processed/predictions.csv`.

## Findings

**Duration and distance carry almost all the signal** (r = 0.95 and 0.94), which
is mechanically unsurprising — both measure how much work was done. Engine power
adds a genuine second axis at r = 0.61.

**Tool width is a trap.** It correlates at −0.66 with consumption, which invites
the conclusion that wider implements are more fuel-efficient. They are not.
Width is constant within two of the four operation types — spraying and
fertilising both use a 24 m boom — so it functions as a proxy for *which
operation this is*, and spraying is short and light while soil work is long and
heavy. Controlling for operation, the relationship largely evaporates. Width
stays in the model as a predictor; it is not a lever anyone can pull.

![Correlation matrix](reports/figures/correlation_matrix.png)

**Residual spread grows with job size**, as expected for a skewed positive
target. Operationally this means an anomaly threshold should be a percentage of
expected consumption, not a fixed number of litres — a fixed threshold would
flag every large job and no small one.

![Holdout performance](reports/figures/holdout_performance.png)

## Limitations

- **No hyperparameter search.** Defaults beat hand-picked values here; a proper
  search would likely beat both. This is the obvious next improvement.
- **500 rows on 30 parcels.** Small enough that the holdout estimate carries
  real uncertainty: six held-out parcels is a small sample, which is why
  cross-validated standard deviations accompany every score.
- **Retrospective only**, for the reason given above.
- **Single season, single operation.** Generalisation to other fleets, crops or
  terrain is untested.

## Layout

```
├── data/
│   ├── raw/            interventions, trajectory sample, parcel outlines
│   └── processed/      merged_data.csv (complete engineered table)
├── notebooks/          1_feature_engineering_and_eda, 2_modeling
├── src/
│   └── data_processing.py    trajectory and parcel feature extraction
├── tests/              pytest suite for the feature pipeline
├── reports/figures/    generated plots
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).
