# EKNet — English Reimplementation — Project Report

**Reference paper:** Liu, Q., Jin, Y., Cao, X., Liu, X., Zhou, X.,
Zhang, Y., Xu, X., & Qi, L. *An Entity Ontology-Based Knowledge Graph
Embedding Approach to News Credibility Assessment*, IEEE Transactions
on Computational Social Systems, 2024 (paper ref `TCSS_1207`).

**Reference repository (idea source for ontology-guided losses):**
[ZihengZZH/OntoEA](https://github.com/ZihengZZH/OntoEA) — *OntoEA:
Ontology-guided Entity Alignment via Joint Knowledge Graph Embedding*,
Findings of ACL-IJCNLP 2021.

This report ties every implementation choice in the repository back to
a specific paper section, table, or figure, so a viva examiner can
verify the correspondence quickly.

---

## 1. Problem statement and scope

Fake news propagates faster than humans can fact-check it. The
reference paper proposes **EKNet** — an end-to-end model that combines
a text encoder with a knowledge-graph encoder and a decision maker to
output (a) a real/fake verdict, and (b) a short comment explaining
*why*.

The paper trains EKNet on a Chinese dataset (**Baidupedia**) plus two
English Kaggle datasets (**Real or Fake**, **Fake News Detection**).
This project re-implements EKNet **in English only**, swapping the
Chinese knowledge base for **Wikidata** and dropping the Chinese word
segmenter for a standard English tokenizer. Everything else
(architecture, training recipe, ablations, comment generation) is
preserved.

## 2. System overview

```
        +----------------+        +-------------------+        +-------------------+
        |   Article x    | -----> |   Text Encoder     | -----> |   text_emb (d=100) |
        |   (English)    |        | (tf-idf weighted   |        +-------------------+
        +----------------+        |  GloVe avg)        |                |
                |                 +-------------------+                |
                |                                                       v
                |  +----------------+  +-------------------+   +-----------------+
                +->| spaCy NER       |->| Wikidata linker    |--> Decision maker -> y_hat
                |  | (mentions)      |  | (cached)           |   (MLP, sigmoid)
                |  +----------------+  +-------------------+
                |                                |
                |                                v
                |                       +-------------------+
                +-----------------------|  KG Encoder        |--> kg_emb (d=64)
                                        |  (TransE +         |
                                        |   membership loss) |
                                        +-------------------+
```

Every block above maps to one folder in [`src/`](../src):
- text encoder → [`src/models/text_encoder.py`](../src/models/text_encoder.py)
- NER → [`src/entity_linking/ner_spacy.py`](../src/entity_linking/ner_spacy.py)
- Wikidata linker → [`src/entity_linking/wikidata_linker.py`](../src/entity_linking/wikidata_linker.py)
- KG encoder → [`src/models/kg_encoder.py`](../src/models/kg_encoder.py)
- TransE → [`src/kg/transE.py`](../src/kg/transE.py)
- Membership loss → [`src/kg/membership_loss.py`](../src/kg/membership_loss.py)
- Decision maker → [`src/models/decision_maker.py`](../src/models/decision_maker.py)
- Full wiring → [`src/models/eknet.py`](../src/models/eknet.py)

## 3. Paper-to-code traceability

| Paper section / figure / table | Code location | Notes |
|---|---|---|
| Sec. III-A — Entity Ontology Framework overview | [`src/ontology/eob_schema.py`](../src/ontology/eob_schema.py) | 4-dim hierarchy (`EoBSchema`) |
| Sec. III-B — EoBSchema + EoBData | [`src/ontology/eob_data.py`](../src/ontology/eob_data.py) | triple store `{(e_i, p_j, v_k)}` |
| Fig. 2 — predefined entity parameters | [`src/ontology/build_schema.py`](../src/ontology/build_schema.py) | populated from Wikidata P31/P279 |
| Sec. IV-A — Text Encoder recipe | [`src/models/text_encoder.py`](../src/models/text_encoder.py) | tf-idf weighted GloVe average |
| Sec. IV-A — KG Encoder | [`src/kg/transE.py`](../src/kg/transE.py) + [`src/models/kg_encoder.py`](../src/models/kg_encoder.py) | TransE + attention-pool |
| Sec. IV-A — Decision Maker | [`src/models/decision_maker.py`](../src/models/decision_maker.py) | MLP over `[text ; kg]` |
| Sec. IV-B — Comment Generator | [`src/comment/rule_based.py`](../src/comment/rule_based.py) | rule + KG-mismatch driven |
| Eq. (1)–(7) — attention / loss | n/a (replaced by rule generator) | see §5 below |
| Sec. IV-C-1 — FastTextRank, Algorithm 1 | [`src/preprocessing/fasttextrank.py`](../src/preprocessing/fasttextrank.py), [`src/preprocessing/augment.py`](../src/preprocessing/augment.py) | English: WordNet ≅ paper's Chinese embedding KNN |
| Sec. IV-C-2 — Chinese word segmentation | [`src/preprocessing/tokenize_en.py`](../src/preprocessing/tokenize_en.py) | replaced by spaCy English tokenizer |
| Sec. V-A-3 — Real or Fake (English) | [`src/data.py`](../src/data.py) (primary) | 6 060 articles, 80/10/10 stratified split |
| Sec. V-A-4 — Fake News Detection (English) | [`src/data.py`](../src/data.py) (held-out) | 3 988 articles |
| Sec. V-B — metrics | [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) | P / R / F1 / Miss-rate + ROUGE |
| Sec. V-C — Adam, lr=1e-4, dropout, L2, early stopping | [`configs/default.yaml`](../configs/default.yaml), [`src/train.py`](../src/train.py) | matches paper |
| Table I — baselines | [`src/baselines/*.py`](../src/baselines) | FastText, TextRNN, TextRCNN, Transformer |
| Table III — ablation modes | [`src/models/eknet.py`](../src/models/eknet.py) (`MODES`) | `text_only`/`ontology_only`/`both` |
| Fig. 5/6 — F1 curves | [`src/evaluation/plots.py`](../src/evaluation/plots.py) | val-loss curves per epoch |

## 4. Differences from the paper, with justification

1. **Knowledge base.** Paper uses **Baidupedia** (Chinese, closed-API).
   We use **Wikidata** via its public MediaWiki API, with a local
   JSONL cache in `data/kb/` for offline replay. Justification: the
   public English datasets in Sec. V-A-3/V-A-4 have no Chinese KB
   coverage; Wikidata covers the entities they reference and is open.

2. **Chinese word segmenter** (Sec. IV-C-2) is unnecessary for English.
   Replaced by spaCy's tokenizer with a regex fallback when spaCy is
   absent. The `DSZM/l` state machine in the paper is purely a Chinese
   construct.

3. **Comment generator** (Sec. IV-B). The paper trains an LSTM
   Encoder-Decoder with attention + PGN + CVG against *reference*
   comments which were curated for the Chinese Baidupedia corpus. No
   such gold comments exist for the English Kaggle datasets. We
   therefore expose a **rule + KG-mismatch driven** generator that
   produces explanations from concrete signals (entity-type
   contradictions, unresolved entities, FastTextRank keywords) and
   evaluate it with ROUGE against article titles as a *proxy*
   reference. This is faithful to the paper's *intent* (interpretable
   comments) while being honest about the absence of gold labels.

4. **Membership loss.** The paper does not formalise an
   ontology-guided embedding loss; it relies on the EoBSchema as
   structural metadata. We borrow OntoEA's `L_M` term to make
   the ontology *actively shape* the entity embeddings — implemented
   in [`src/kg/membership_loss.py`](../src/kg/membership_loss.py) and
   applied post-hoc after TransE training (configurable weight).

5. **Compute.** All models are sized for **CPU-only** training:
   GloVe-100d (instead of 300d or BERT), TransE-64d (instead of
   higher), 2-layer 4-head Transformer (instead of a stack of 6+),
   maximum 400 tokens per article. This compresses the paper's
   experiments into a few hours on a laptop without changing the
   experimental story.

## 5. Implementation notes

### 5.1 Entity Ontology Framework (`src/ontology/`)

`EoBSchema` is a 4-dimensional class hierarchy (Fig. 2). Each
top-level dimension is a spaCy NER type (`PERSON`, `ORG`, `GPE`, …);
its children come from Wikidata P31/P279 chains. The lookup
`crossview[qid] = "PERSON.Q5.Q729146"` is the English analogue of
OntoEA's `crossview_link_1`.

`EoBData` stores `(entity, relation, value)` triples plus three
indices (`by_entity`, `by_class`, `crossview`) for O(1) traversal —
exactly the triple-store described in Sec. III-B.

### 5.2 Text encoder (`src/models/text_encoder.py`)

Implements the paper's exact recipe:

```
sentence_emb_j = sum_w tfidf(w) * glove(w) / sum_w tfidf(w)
news_emb       = sum_j w_j * sentence_emb_j / sum_j w_j
```

`w_j` is the sum of token tf-idf weights in sentence j (the paper's
"weighted average over sentence embeddings").

### 5.3 KG encoder (`src/kg/transE.py`, `src/models/kg_encoder.py`)

* TransE via **PyKEEN** when available, with a 60-line PyTorch
  fallback otherwise (so the repo runs anywhere).
* Per-article KG = entities found via NER + their P31/P279/MEMBER_OF
  triples.
* Article-level KG embedding = tf-idf-weighted mean of entity
  vectors (attention-pool).

### 5.4 Decision maker (`src/models/decision_maker.py`)

`Linear(text+kg) -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> sigmoid`.
Loss = BCE-with-logits. Optimiser = Adam, `lr=1e-4`,
`weight_decay=1e-5` (matches paper Sec. V-C). Early stopping with
patience=5 on validation loss (matches paper).

### 5.5 Ablations (`src/models/eknet.py::MODES`)

Same model, three runs:
* `text_only`: zero out kg_emb at forward time → reproduces row 2 of
  paper Table III.
* `ontology_only`: zero out text_emb → reproduces row 3.
* `both`: full EKNet → reproduces row 1.

### 5.6 Comment generator (`src/comment/rule_based.py`)

Steps:
1. Run NER over the article.
2. For each mention, link to a Wikidata QID and fetch P31.
3. If the NER label (e.g. `PERSON`) is incompatible with the
   Wikidata superclass (e.g. P31 points to `Q838948` = work-of-art),
   flag the mention.
4. Pull the top-3 FastTextRank keywords.
5. Compose a 1-2 sentence comment.

Evaluation uses ROUGE-1/2/L against the article title (Sec. V-B's
metric set), with the caveat that title-vs-comment ROUGE is a noisy
proxy for the paper's curated reference comments.

## 6. How to reproduce

Step-by-step commands are in [`README.md`](../README.md). Summary:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4 punkt
python scripts/download_data.py
python scripts/download_glove.py
python -m src.entity_linking.kb_cache --warm
python -m src.ontology.build_schema
python -m src.train --model eknet --mode text_only
python -m src.train --model eknet --mode ontology_only
python -m src.train --model eknet --mode both
python -m src.train --model textrnn
python -m src.train --model textrcnn
python -m src.train --model transformer_small
python -m src.train --model fasttext
python -m src.evaluate --emit-tables
streamlit run app/streamlit_app.py
```

Tables I & III land in `reports/`, the Fig. 5/6-analogue plot lands in
`reports/figures/`, and the Streamlit demo lets you paste an article
to see the verdict + comment in real time.

## 7. Limitations and future work

* **Comment ROUGE proxy** — see §4-3 above. With a small labelling
  effort (or a separate weak-supervision pass), one could generate
  proper reference comments and re-train the BiLSTM+LSTM+PGN+CVG
  generator from the paper. Hooks for self-service sample generation
  are already in [`src/preprocessing/augment.py`](../src/preprocessing/augment.py).
* **Wikidata coverage** — entities not in Wikidata silently fall
  through to zero vectors. A future improvement is to back off to
  DBpedia or to a fuzzy-string nearest-neighbour over QIDs.
* **Scale** — the implementation comfortably handles ~10 K articles
  on CPU. For "big data"-scale corpora (Sec. I), swap the in-memory
  KB cache for a sqlite-backed one (interface is already abstracted in
  [`src/entity_linking/kb_cache.py`](../src/entity_linking/kb_cache.py)).

## 8. References

1. Liu, Q. et al. *An Entity Ontology-Based Knowledge Graph Embedding
   Approach to News Credibility Assessment.* IEEE TCSS, 2024.
2. Xiang, Y. et al. *OntoEA: Ontology-guided Entity Alignment via
   Joint Knowledge Graph Embedding.* Findings of ACL-IJCNLP, 2021.
3. Pennington, J., Socher, R., & Manning, C. *GloVe: Global Vectors
   for Word Representation.* EMNLP, 2014.
4. Bordes, A. et al. *Translating Embeddings for Modeling
   Multi-relational Data.* NIPS, 2013.
5. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS, 2017.
6. Vrandečić, D. & Krötzsch, M. *Wikidata: a free collaborative
   knowledgebase.* CACM, 2014.
