# WeSTeC — Weakly Supervised Fake News Classification

Code for **early detection of fake news on emerging topics** using weak supervision (WeSTeC: Weakly Supervised Text Classification).

The framework programmatically generates labeling functions (content heuristics + classical ML models), aggregates them with Snorkel / HyperLabelModel, and trains a RoBERTa classifier on the resulting weak labels—useful when ground-truth labels are scarce for a new topic.

## Publications

- **Paper:** [Early detection of fake news on emerging topics through weak supervision](https://link.springer.com/article/10.1007/s10844-024-00852-1)  
  *Journal of Intelligent Information Systems*, 2024. DOI: [10.1007/s10844-024-00852-1](https://doi.org/10.1007/s10844-024-00852-1)
- **Thesis:** [METU open archive](https://open.metu.edu.tr/handle/11511/105390) (Serhat Hakkı Akdağ, M.Sc., Middle East Technical University, 2023)

If you use this repository, please cite the paper (see [`CITATION.cff`](CITATION.cff)).

```bibtex
@article{akdag2024early,
  title   = {Early detection of fake news on emerging topics through weak supervision},
  author  = {Akda{\u{g}}, Serhat Hakk{\i} and Cicekli, Nihan Kesim},
  journal = {Journal of Intelligent Information Systems},
  volume  = {62},
  pages   = {1263--1284},
  year    = {2024},
  doi     = {10.1007/s10844-024-00852-1}
}
```

## Repository layout

```
configs/                      # Curated labeling-function sets used in the paper
content_feature_thresholds/   # Precomputed content-feature thresholds per dataset
data/                         # NELA inputs + intermediate CSVs (not shipped; see below)
model/                        # Trained classical models / TF-IDF pickles (local)
notebooks/
  analysis/                   # Figures, exploration, weak-label analysis
  experiments/                # End-model and LF-model experiments
  figures/                    # Cached stats used by figure notebooks
src/
  preprocessing/              # NELA loading, cleanup, sampling, TF-IDF
  weak_labeling/              # Features, thresholds, LFs, aggregation
  end_models/                 # RoBERTa end classifier
```

## Pipeline overview

1. **Prepare NELA data** — sample/clean articles, attach source reliability labels  
2. **Inject content features** — stylistic, POS, readability features via spaCy  
3. **Select thresholds** — turn feature distributions into content labeling functions  
4. **Train classical LF models** — TF-IDF + XGBoost / LR / NB / RF / AdaBoost  
5. **Inject model predictions** as additional labeling functions  
6. **Aggregate weak labels** — Snorkel majority vote, Snorkel label model, HyperLabelModel  
7. **Train end model** — fine-tune RoBERTa on weak (or gold) labels

## Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate wastec

# Or pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# spaCy English model (required for content features)
python -m spacy download en_core_web_sm
```

Run modules from the **repository root** so `src.*` imports resolve:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Data (NELA)

Raw NELA SQLite dumps are **not** included (size + redistribution terms). Place them under:

```
data/raw/nela/nela-covid-2020.db
data/raw/nela/nela-covid-2021.db
data/raw/nela/nela-elections-2020.db
data/raw/nela/nela-capitol-2021.db
data/raw/nela/source-labels.csv
```

NELA releases are available from the [NELA project](https://github.com/BenjaminDHorne/NELA-GT) / associated dataset pages. Intermediate CSVs (`data/feature_injected/`, `data/with_weak_labels/`) and pickles under `model/` are produced by the steps below and are gitignored.

Tracked reference artifacts:

- `content_feature_thresholds/*.json` — thresholds used to build content LFs  
- `data/with_weak_labels/runs.json` — logged weak-labeler metrics from paper runs  
- `configs/curated_labeling_functions.json` — curated LF names from the paper  

## Reproducing the main steps

Each stage is a runnable module (`python -m` style via file path). Typical order:

```bash
# 1) Sample / combine / subset NELA articles (edit __main__ configs as needed)
python src/preprocessing/read_raw_data.py

# 2) Content feature injection
python src/weak_labeling/content_based/feature_injection/content_feature_injection.py

# 3) Fit shared TF-IDF vectorizers
python src/preprocessing/tfidf_vectorizer.py

# 4) Train classical models used as labeling functions
python src/weak_labeling/model_based/xgboost/xgboost_model.py
python src/weak_labeling/model_based/logistic_regression/logistic_regression.py
python src/weak_labeling/model_based/naive_bayes/naive_bayes.py
python src/weak_labeling/model_based/random_forest/random_forest.py
python src/weak_labeling/model_based/adaboost/adaboost.py

# 5) Threshold selection for content LFs
python src/weak_labeling/content_based/threshold_selection/threshold_selection.py

# 6) Inject model-based features into target datasets
python src/weak_labeling/model_based/labeling_functions/model_feature_injection.py

# 7) Aggregate weak labels (uses configs/curated_labeling_functions.json)
python src/weak_labeling/weak_labeler/weak_labeler.py

# 8) RoBERTa end model (GPU recommended)
python src/end_models/roberta_classifier.py
# or: python src/end_models/simpletransformers.py
```

Notebooks under `notebooks/analysis/` regenerate paper figures and analyze `runs.json`. Experiment notebooks under `notebooks/experiments/` cover LF-model and end-model trials.

## Notes

- Labels: `0 = real`, `1 = fake`, `-1 = abstain` (Snorkel convention).  
- Pickled models were trained with scikit-learn ~1.1 / XGBoost ~1.5; newer versions may not load old pickles—retrain if needed.  
- End-model training defaults to CUDA (`use_cuda=True`); pass `use_cuda=False` from `run_paper_end_model` on CPU-only machines.  
- This is research code cleaned for release; expect long runtimes on full NELA corpora.

## License

MIT — see [`LICENSE`](LICENSE).
