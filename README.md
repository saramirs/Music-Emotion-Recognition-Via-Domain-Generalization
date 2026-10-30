# Music Emotion Recognition via Domain Generalization: Aligning Emotions, Not Datasets

Code for my MSc thesis (Computer Science, Università degli Studi di Milano).

The question: when a model that predicts the **valence** (how positive) and **arousal** (how energetic) of a song is tested on a dataset it has **never seen**, does it generalize better if it aligns **emotions** across the training datasets, or if it aligns the **datasets** themselves?

Three methods are compared, all on top of the same frozen music model (MERT-v1-95M) and the same small trained network:

| Method | Training loss |
|---|---|
| Baseline | MSE |
| Domain alignment (DANN) | MSE + λ × domain loss (a domain classifier with gradient reversal) |
| Emotion alignment (proposed) | MSE + w × emotion loss (songs with similar ratings get similar embeddings, across datasets) |

## Notebooks

Run them in this order on Google Colab (GPU, e.g. NVIDIA L4). Paths point to a Google Drive project folder; change `DRIVE_PROJECT` / `DATASETS_DIR` at the top of each notebook to your own location.

| Order | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/Data_Preparation.ipynb` | Reads the five datasets, cleans them, rescales all ratings to 0–1, and extracts frozen MERT-v1-95M features (13 layers × 768 per 5-second window). |
| 2 | `notebooks/Training_Three_Folds.ipynb` | Development run of the leave-one-dataset-out protocol (three folds). Its findings fixed some settings of the final experiment. |
| 3 | `notebooks/Final_Experiment.ipynb` | Final, pre-declared experiment: three folds and a locked final test on Soundtracks, 10 seeds per method, strengths chosen on validation songs only, a lock file written before any test prediction, paired comparisons and bootstrap intervals. |

The notebooks share data only through the output folder of the first one (`MER_DG_v2` on Drive); a few internal messages still refer to it as "Notebook 16", the name it had when the experiments were run.

## Protocol in short

* Domains: MediaEval (DEAM + EmoMusic), Memo2496 and PMEmo; each fold trains on two and tests on the third. Soundtracks is used only once, as the final test (trained on all three).
* Every choice (best epoch, λ, w) is made on validation songs of the training datasets (20 % of each), never on test songs.
* Main metric: **macro CCC** (mean of the valence CCC and the arousal CCC). Pearson, MAE and RMSE are reported for reference.
* Each method is trained with 10 random seeds; the reported result is the mean of the 10 models.

## Results

`results/` holds the final numbers produced by the Final Experiment notebook:

| File | Content |
|---|---|
| `final_results_table.csv` | Mean and SD over 10 seeds, per test set and method, for every metric |
| `final_results_per_seed.csv` | The same metrics for every seed |
| `final_seed_ensemble.csv` | Metrics when the 10 seeds' predictions are averaged per song |

Macro CCC on the unseen test sets (mean of 10 seeds):

| Test set | Baseline | Domain alignment | Emotion alignment |
|---|---|---|---|
| Soundtracks (final test) | 0.385 | 0.411 | **0.431** |
| Memo2496 | 0.559 | 0.567 | **0.583** |
| PMEmo (exploratory) | 0.604 | 0.586 | **0.609** |
| MediaEval | **0.662** | 0.661 | 0.655 |

## Data

The datasets are not included here; please obtain them from their original sources under their own licences: DEAM, EmoMusic (MediaEval), PMEmo, Memo2496 and the Soundtracks dataset (Eerola and Vuoskoski, 2011). The pretrained model is `m-a-p/MERT-v1-95M` from Hugging Face.

## Requirements

See `requirements.txt` (Google Colab already provides most packages).
