# EKNet — English Reimplementation

A from-scratch, CPU-friendly **English** reimplementation of:

> Liu, Q., Jin, Y., Cao, X., Liu, X., Zhou, X., Zhang, Y., Xu, X., & Qi, L.
> *"An Entity Ontology-Based Knowledge Graph Embedding Approach to News
> Credibility Assessment"* (IEEE TCSS, 2024) — paper ref `TCSS_1207`.

The original paper uses **Baidupedia** (Chinese) and a Chinese word
segmenter. This reimplementation swaps Baidupedia for **Wikidata** and
runs entirely on English text, while keeping the architectural pieces
of EKNet intact:

- An **Entity Ontology Framework** (`EoBSchema` + `EoBData`)
- A **Text Encoder** (tf-idf weighted GloVe sentence embeddings)
- A **Knowledge Graph Encoder** (TransE via PyKEEN + OntoEA-style
  membership loss)
- A **Decision Maker** (MLP over `[text ; kg]`)
- A **Comment Generator** (rule + KG-mismatch driven, evaluated with
  ROUGE against article titles as a proxy reference)

---

## Paper section → code map

| Paper section | Code |
|---|---|
| Sec. III — Entity Ontology Framework | [`src/ontology/`](src/ontology) |
| Sec. IV-A — Text Encoder, KG Encoder, Decision Maker | [`src/models/`](src/models) |
| Sec. IV-B — Comment Generator | [`src/comment/`](src/comment) |
| Sec. IV-C — Data Enhancement, FastTextRank, Coverage | [`src/preprocessing/`](src/preprocessing) |
| Sec. V-A — Datasets | [`data/raw/`](data/raw), [`scripts/download_data.py`](scripts/download_data.py) |
| Sec. V-B — Metrics | [`src/evaluation/metrics.py`](src/evaluation/metrics.py) |
| Sec. V-D — Table I baselines | [`src/baselines/`](src/baselines) |
| Sec. V-E — Table III ablations | `--mode {text_only,ontology_only,both}` in [`src/train.py`](src/train.py) |
| Figs. 5–6 — F1 plots | [`src/evaluation/plots.py`](src/evaluation/plots.py) |

---

## Datasets

Primary (used in paper, Sec. V-A-3):
* **Real or Fake** (Kaggle, `rchitic17/real-or-fake`) — 6 060 articles,
  balanced 50/50. Schema: `id, title, text, label`.

Secondary held-out (used in paper, Sec. V-A-4):
* **Fake News Detection** (Kaggle, `jruvika/fake-news-detection`) —
  3 988 articles. Schema: `URLs, Headline, Body, Label`.

```bash
# requires `kaggle` CLI configured with API key
python scripts/download_data.py
```

If you cannot use the Kaggle CLI, manually drop the CSVs at:
```
data/raw/real_or_fake/fake_or_real_news.csv
data/raw/fake_news_detection/data.csv
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                    # (PowerShell: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4 punkt
```

GloVe embeddings (100d) — download once:
```bash
python scripts/download_glove.py
```

### PyKEEN install fix (Python 3.12 / fresh venvs)

`requirements.txt` pins `pykeen==1.10.2`, but on a clean Python 3.12 venv
two of its transitive dependencies are too new and break the import
chain. Symptoms you may see after a plain `pip install -r requirements.txt`:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

or, after fixing that:

```text
TypeError: Too few arguments for <class 'class_resolver.func.FunctionResolver'>;
actual 1, expected at least 2
```

Run these two pins to fix both at once **before** the first training run:

```powershell
pip install "setuptools<70" "class-resolver<0.5"
```

Why:

* `pykeen 1.10.2` imports `from pkg_resources import iter_entry_points`,
  which is part of the **setuptools** package. Setuptools 70+ removed
  `pkg_resources`; setuptools 60-69 still ships it.
* `pykeen 1.10.2` declares `FunctionResolver[Initializer]` (one type
  parameter). **class-resolver** 0.5+ made `FunctionResolver` take two
  type parameters, which makes PyKEEN's import fail. `class-resolver<0.5`
  keeps the old signature.

Verify both pins work with PyKEEN's official toy dataset:

```powershell
python -c "from pykeen.pipeline import pipeline; from pykeen.datasets import Nations; r = pipeline(dataset=Nations, model='TransE', training_kwargs={'num_epochs': 2}, model_kwargs={'embedding_dim': 16}); print('PyKEEN OK')"
```

You should see a tqdm progress bar finish and then `PyKEEN OK`.

Once PyKEEN is importable, `src/kg/transE.py` will use it automatically;
the log line changes from `transE-native ep N/M` to PyKEEN's
`Training epochs on cpu: ... epoch/s` output. If you have an existing
feature cache from a previous (native-TransE) run, force a rebuild so
PyKEEN's embeddings are written instead:

```powershell
python -m src.train --model eknet --mode both --profile balanced --rebuild-features
```

**Don't want to deal with the pins?** PyKEEN is optional. The pipeline
falls back to a tiny pure-PyTorch TransE in `src/kg/transE.py` with the
same interface; you'll just get slightly slower KG embedding training
and no link-prediction metrics. Either path produces the EKNet
features the model needs.

### MiniLM text encoder (optional, used by `--profile accuracy`)

The `text_encoder.type: minilm` path (and `python -m src.train ... --profile accuracy`)
loads `sentence-transformers/all-MiniLM-L6-v2` for stronger sentence
embeddings (~+5-8 F1 on English news). The catch: very recent releases
of `sentence-transformers` and `transformers` require **PyTorch >= 2.4**
and crash mid-import with

```text
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.3.1
NameError: name 'nn' is not defined
```

The fix is to pin a torch-2.3-compatible trio of packages:

```powershell
pip uninstall -y sentence-transformers transformers
pip install "sentence-transformers==2.7.0" "transformers>=4.39,<4.45" "huggingface_hub<0.24"
```

Verify:

```powershell
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('MiniLM OK, dim =', m.get_sentence_embedding_dimension())"
```

Should print `MiniLM OK, dim = 384`. The model (~80 MB) is downloaded
once into `%USERPROFILE%\.cache\huggingface\` and reused thereafter.

These pins are listed (commented out) at the bottom of `requirements.txt`.
Uncomment them if you want `pip install -r requirements.txt` to set up
MiniLM automatically on a fresh venv.

If you just want the paper-faithful EKNet (the GloVe text encoder),
ignore this section entirely — `--profile balanced` or the defaults
work without `sentence-transformers`.

---

## Reproducing the paper's tables

```bash
# 1) cache Wikidata for every entity in the corpus (one-time, slow)
python -m src.entity_linking.kb_cache --warm

# 2) train EKNet with all three ablations
python -m src.train --mode text_only
python -m src.train --mode ontology_only
python -m src.train --mode both

# 3) train baselines (Table I)
python -m src.train --model fasttext
python -m src.train --model textrnn
python -m src.train --model textrcnn
python -m src.train --model transformer_small

# 4) collect everything into Markdown tables + plots
python -m src.evaluate --emit-tables
```

Tables I & III and the Fig. 5/6-style plots land in `reports/`.

---

## Demo

```bash
streamlit run app/streamlit_app.py
```

Paste an article → see real/fake verdict, highlighted entities, the
generated comment, and an interactive entity-relation graph.

---

## Repo layout

See [the project plan](.cursor/plans) for a longer explanation; the
folder tree is roughly:

```
final_sem_project/
  configs/default.yaml
  data/{raw,processed,kb,embeddings}/
  src/
    preprocessing/   ontology/   entity_linking/
    kg/   models/   comment/   baselines/   evaluation/
    train.py   evaluate.py
  scripts/           # one-off helpers (data + embeddings download)
  notebooks/         # EDA, ontology walkthrough, results
  app/streamlit_app.py
  reports/           # auto-generated tables + figures
```

---

## Citing the original paper

```bibtex
@article{Liu2024EKNet,
  author  = {Liu, Qi and Jin, Yuanyuan and Cao, Xuefei and Liu, Xiaodong and
             Zhou, Xiaokang and Zhang, Yonghong and Xu, Xiaolong and Qi, Lianyong},
  title   = {An Entity Ontology-Based Knowledge Graph Embedding Approach to
             News Credibility Assessment},
  journal = {IEEE Transactions on Computational Social Systems},
  year    = {2024},
  doi     = {10.1109/TCSS.2024.10431771}
}
```

OntoEA (which inspired the membership-loss term we borrow):

```bibtex
@inproceedings{xiang-etal-2021-ontoea,
  title     = {OntoEA: Ontology-guided Entity Alignment via Joint Knowledge Graph Embedding},
  author    = {Xiang, Yuejia and Zhang, Ziheng and Chen, Jiaoyan and Chen, Xi and Lin, Zhenxi and Zheng, Yefeng},
  booktitle = {Findings of ACL-IJCNLP 2021},
  year      = {2021}
}
```
