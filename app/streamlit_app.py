"""Streamlit demo for the EKNet English reimplementation.

Run:
    streamlit run app/streamlit_app.py

Tabs
----
1. Assess Article  -- single-model EKNet prediction + entities + comment
2. Compare Models  -- run every available model on the pasted article
3. Model Metrics   -- table + bar chart of every trained model's report,
                      plus val-loss curves for the neural ones

Artifacts the demo will use if present:
    * artifacts/eknet_{both,text_only,ontology_only}.pt
    * artifacts/textrnn.pt + textrnn_vocab.json
    * artifacts/textrcnn.pt + textrcnn_vocab.json
    * artifacts/transformer_small.pt + transformer_small_vocab.json
    * artifacts/fasttext.bin (native) or fasttext.pkl (fallback)
    * artifacts/feature_cache/{tfidf.pkl, transE.npz}
    * artifacts/*_report.json and *_history.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# allow imports from project root when invoked via streamlit run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import load_config
from src.preprocessing.tokenize_en import tokenize_article
from src.preprocessing.tfidf import TfIdfWeights
from src.models.text_encoder import GloveLoader, TextEncoder
from src.models.kg_encoder import KGEncoder
from src.models.eknet import EKNet
from src.kg.transE import KGEmbeddings
from src.entity_linking.kb_cache import KBCache
from src.entity_linking.ner_spacy import extract_mentions
from src.entity_linking.wikidata_linker import WikidataLinker
from src.comment.rule_based import generate_comment


ARTIFACTS = Path("artifacts")
FEAT_CACHE = ARTIFACTS / "feature_cache"


st.set_page_config(page_title="EKNet -- News Credibility", layout="wide")


# ----------------------------------------------------------------------
#  cached resource loaders
# ----------------------------------------------------------------------

@st.cache_resource
def _load_core():
    """Encoders + linker that are shared across every tab."""
    cfg = load_config()
    glove = GloveLoader(cfg.paths.glove, dim=cfg.text_encoder.glove_dim)

    tfidf_path = FEAT_CACHE / "tfidf.pkl"
    if not tfidf_path.exists():
        tfidf_path = ARTIFACTS / "tfidf.pkl"
    tfidf = TfIdfWeights.load(tfidf_path) if tfidf_path.exists() else None
    text_enc = TextEncoder(glove, tfidf_weights=tfidf)

    kg_path = FEAT_CACHE / "transE.npz"
    if not kg_path.exists():
        kg_path = ARTIFACTS / "transE.npz"
    if kg_path.exists():
        kg_emb = KGEmbeddings.load(kg_path)
    else:
        kg_emb = KGEmbeddings(
            entity_to_idx={},
            relation_to_idx={},
            entity_emb=np.zeros((1, cfg.kg.embedding_dim), dtype=np.float32),
            relation_emb=np.zeros((1, cfg.kg.embedding_dim), dtype=np.float32),
        )
    kg_enc = KGEncoder(kg_emb)

    cache = KBCache(cfg.paths.kb_cache)
    linker = WikidataLinker(
        cache,
        endpoint=cfg.entity_linking.wikidata_endpoint,
        user_agent=cfg.entity_linking.wikidata_user_agent,
        top_relations=cfg.entity_linking.top_relations,
    )
    return cfg, text_enc, kg_enc, linker


@st.cache_resource
def _load_eknet(mode: str):
    """Load a specific EKNet ablation checkpoint."""
    cfg, *_ = _load_core()
    ckpt = ARTIFACTS / f"eknet_{mode}.pt"
    if not ckpt.exists():
        return None
    model = EKNet(
        text_dim=cfg.text_encoder.glove_dim,
        kg_dim=cfg.kg.embedding_dim,
        mode=mode,
        hidden_dims=tuple(cfg.model.hidden_dims),
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def _load_baseline_torch(name: str):
    """Load TextRNN/TextRCNN/Transformer model + vocab from artifacts."""
    cfg, *_ = _load_core()
    ckpt = ARTIFACTS / f"{name}.pt"
    vpath = ARTIFACTS / f"{name}_vocab.json"
    if not (ckpt.exists() and vpath.exists()):
        return None

    from src.baselines._common import load_vocab

    vocab, max_len = load_vocab(vpath)

    if name == "textrnn":
        from src.baselines.textrnn import TextRNN as Cls
        model = Cls(vocab_size=len(vocab), embed_dim=cfg.text_encoder.glove_dim,
                    hidden_dim=cfg.baselines.hidden_dim)
    elif name == "textrcnn":
        from src.baselines.textrcnn import TextRCNN as Cls
        model = Cls(vocab_size=len(vocab), embed_dim=cfg.text_encoder.glove_dim,
                    hidden_dim=cfg.baselines.hidden_dim)
    elif name == "transformer_small":
        from src.baselines.transformer_small import SmallTransformer as Cls
        model = Cls(vocab_size=len(vocab), embed_dim=cfg.text_encoder.glove_dim,
                    max_len=max_len)
    else:
        return None
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return {"name": name, "model": model, "vocab": vocab, "max_len": max_len}


@st.cache_resource
def _load_fasttext():
    """Native fasttext.bin if available, else sklearn fallback fasttext.pkl."""
    bin_path = ARTIFACTS / "fasttext.bin"
    pkl_path = ARTIFACTS / "fasttext.pkl"
    if bin_path.exists():
        try:
            import fasttext
            m = fasttext.load_model(str(bin_path))
            return {"kind": "native", "model": m}
        except Exception:
            pass
    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            return {"kind": "sklearn", "model": pickle.load(f)}
    return None


def _list_available_models() -> dict[str, bool]:
    """Which models are actually trained right now?"""
    out: dict[str, bool] = {}
    for m in ("both", "text_only", "ontology_only"):
        out[f"eknet_{m}"] = (ARTIFACTS / f"eknet_{m}.pt").exists()
    for n in ("textrnn", "textrcnn", "transformer_small"):
        out[n] = (ARTIFACTS / f"{n}.pt").exists() and (ARTIFACTS / f"{n}_vocab.json").exists()
    out["fasttext"] = (ARTIFACTS / "fasttext.bin").exists() or (ARTIFACTS / "fasttext.pkl").exists()
    return out


# ----------------------------------------------------------------------
#  inference helpers
# ----------------------------------------------------------------------

def _encode_article_inputs(text: str, cfg, text_enc: TextEncoder, kg_enc: KGEncoder,
                           linker: WikidataLinker):
    """Tokenise + extract entities + build (text_vec, kg_vec, entity_rows)."""
    sents = tokenize_article(
        text,
        spacy_model=cfg.preprocessing.spacy_model,
        max_sentences=cfg.preprocessing.max_sentences,
        max_tokens_per_sentence=cfg.preprocessing.max_tokens_per_sentence,
    )
    tv = text_enc.encode_article(sents)

    mentions = extract_mentions(text, model=cfg.entity_linking.spacy_ner_model)
    qids: list[str] = []
    ent_rows = []
    for m in mentions[: cfg.entity_linking.max_entities_per_article]:
        ent = linker.link(m.text)
        if ent.qid:
            qids.append(ent.qid)
        ent_rows.append({
            "mention": m.text,
            "type": m.label,
            "qid": ent.qid or "",
            "label": ent.label or "",
            "description": ent.description or "",
        })
    kv = kg_enc.encode(qids)
    return tv, kv, ent_rows


def _predict_eknet(model, tv: np.ndarray, kv: np.ndarray) -> float:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(tv).float().unsqueeze(0),
            torch.from_numpy(kv).float().unsqueeze(0),
        )
        return float(torch.sigmoid(logits).item())


def _predict_torch_baseline(b, text: str, cfg) -> float:
    from src.baselines._common import encode_text_for_inference
    ids = encode_text_for_inference(text, b["vocab"], b["max_len"],
                                    spacy_model=cfg.preprocessing.spacy_model)
    x = ids.unsqueeze(0)  # (1, L)
    lens = torch.tensor([ids.shape[0]], dtype=torch.long)
    with torch.no_grad():
        logits = b["model"](x, lens)
        return float(torch.sigmoid(logits).item())


def _predict_fasttext(ft, text: str) -> float:
    text_clean = " ".join(text.split())
    if ft["kind"] == "native":
        labels, probs = ft["model"].predict(text_clean, k=2)
        for lbl, p in zip(labels, probs):
            if lbl == "__label__fake":
                return float(p)
        return 0.0
    return float(ft["model"].predict_proba([text_clean])[0, 1])


# ----------------------------------------------------------------------
#  reports + history loading (for the metrics tab)
# ----------------------------------------------------------------------

def _load_all_reports() -> pd.DataFrame:
    rows = []
    for p in sorted(ARTIFACTS.glob("*_report.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_all_histories() -> dict[str, list[dict]]:
    out = {}
    for p in sorted(ARTIFACTS.glob("*_history.json")):
        name = p.stem.replace("_history", "")
        try:
            with open(p, "r", encoding="utf-8") as f:
                h = json.load(f)
            if isinstance(h, dict) and "history" in h:
                h = h["history"]
            out[name] = h
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------
#  Tab 1: Assess
# ----------------------------------------------------------------------

def tab_assess() -> None:
    cfg, text_enc, kg_enc, linker = _load_core()
    available = _list_available_models()
    eknet_modes = [m for m in ("both", "text_only", "ontology_only") if available[f"eknet_{m}"]]
    if not eknet_modes:
        st.error("No trained EKNet checkpoint found. Run `python -m src.train --model eknet --mode both` first.")
        return

    mode = st.selectbox("EKNet ablation", eknet_modes, index=0,
                        help="Pick which trained EKNet variant to use for the verdict.")
    text = st.text_area("Paste a news article:", height=280, placeholder="Article text ...",
                        key="assess_text")
    if not text.strip():
        st.info("Paste an article above and click 'Assess'.")
        return

    if not st.button("Assess credibility", key="assess_btn"):
        return

    with st.spinner("Running entity linking + EKNet ..."):
        tv, kv, ent_rows = _encode_article_inputs(text, cfg, text_enc, kg_enc, linker)
        model = _load_eknet(mode)
        prob = _predict_eknet(model, tv, kv) if model is not None else None
        verdict = "FAKE" if (prob or 0) >= 0.5 else "REAL"
        comment = generate_comment(
            text, linker=linker, ner_model=cfg.entity_linking.spacy_ner_model,
            top_keywords=cfg.comment_generator.top_keywords,
        )

    col_v, col_c = st.columns([1, 2])
    with col_v:
        st.metric("Verdict", verdict)
        if prob is not None:
            st.progress(min(max(prob, 0.0), 1.0), text=f"Fake probability: {prob:.2%}")
        st.caption(f"Model: EKNet ({mode})")
    with col_c:
        st.markdown("**Generated comment**")
        st.write(comment.text)
        if comment.flagged_reasons:
            with st.expander("Flag details"):
                for r in comment.flagged_reasons:
                    st.write("- " + r)

    st.markdown("### Extracted entities")
    if ent_rows:
        st.dataframe(ent_rows, use_container_width=True)
        try:
            _render_pyvis(ent_rows)
        except Exception as e:
            st.warning(f"Entity graph rendering failed: {e}")
    else:
        st.info("No named entities found.")


# ----------------------------------------------------------------------
#  Tab 2: Compare every available model
# ----------------------------------------------------------------------

def tab_compare() -> None:
    cfg, text_enc, kg_enc, linker = _load_core()
    available = _list_available_models()
    have_any = any(available.values())
    st.markdown("Run **every trained model** on the same article and see how they disagree.")
    with st.expander("Available checkpoints", expanded=False):
        st.json(available)
    if not have_any:
        st.error("No trained models found in artifacts/. Train at least one with `python -m src.train ...`.")
        return

    text = st.text_area("Paste a news article:", height=240, placeholder="Article text ...",
                        key="compare_text")
    if not text.strip():
        st.info("Paste an article above and click 'Compare'.")
        return
    if not st.button("Compare models", key="compare_btn"):
        return

    rows = []
    with st.spinner("Encoding article once ..."):
        tv, kv, ent_rows = _encode_article_inputs(text, cfg, text_enc, kg_enc, linker)

    # EKNet variants
    for mode in ("both", "text_only", "ontology_only"):
        if not available[f"eknet_{mode}"]:
            continue
        m = _load_eknet(mode)
        if m is None:
            continue
        p = _predict_eknet(m, tv, kv)
        rows.append({"model": f"EKNet ({mode})", "fake_prob": p, "verdict": "FAKE" if p >= 0.5 else "REAL"})

    # Neural baselines
    for name in ("textrnn", "textrcnn", "transformer_small"):
        if not available[name]:
            continue
        b = _load_baseline_torch(name)
        if b is None:
            continue
        with st.spinner(f"Running {name} ..."):
            p = _predict_torch_baseline(b, text, cfg)
        rows.append({"model": name, "fake_prob": p, "verdict": "FAKE" if p >= 0.5 else "REAL"})

    # FastText
    if available["fasttext"]:
        ft = _load_fasttext()
        if ft is not None:
            p = _predict_fasttext(ft, text)
            rows.append({"model": "fasttext", "fake_prob": p, "verdict": "FAKE" if p >= 0.5 else "REAL"})

    if not rows:
        st.warning("No model produced a prediction (check the checkpoints).")
        return

    df = pd.DataFrame(rows)
    df_display = df.copy()
    df_display["fake_prob"] = df_display["fake_prob"].map(lambda v: f"{v:.2%}")
    st.markdown("### Per-model verdict")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("### Fake-probability comparison")
    chart_df = df.set_index("model")[["fake_prob"]]
    st.bar_chart(chart_df, height=320)

    # consensus
    n_fake = int((df["fake_prob"] >= 0.5).sum())
    n_total = len(df)
    consensus = "FAKE" if n_fake > n_total / 2 else "REAL"
    st.metric("Majority verdict",
              consensus,
              delta=f"{n_fake}/{n_total} models say FAKE")

    if ent_rows:
        with st.expander("Entities found in this article"):
            st.dataframe(ent_rows, use_container_width=True)


# ----------------------------------------------------------------------
#  Tab 3: Metrics dashboard
# ----------------------------------------------------------------------

def tab_metrics() -> None:
    df = _load_all_reports()
    if df.empty:
        st.warning(
            "No `*_report.json` files in `artifacts/`. "
            "Run `python -m src.train --model <m> ...` and re-open this tab."
        )
        return

    metric_cols = [c for c in ("precision", "recall", "f1", "miss_rate", "auc") if c in df.columns]
    pretty = df[["model"] + metric_cols].copy()
    for c in metric_cols:
        pretty[c] = pretty[c].astype(float).round(4)
    pretty = pretty.sort_values("f1", ascending=False) if "f1" in pretty.columns else pretty

    st.markdown("### Test-set metrics (from `artifacts/*_report.json`)")
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    if "f1" in df.columns:
        st.markdown("### F1 across models")
        st.bar_chart(pretty.set_index("model")[["f1"]], height=300)

    long_cols = [c for c in ("precision", "recall", "f1") if c in df.columns]
    if long_cols:
        st.markdown("### Precision / Recall / F1 by model")
        long_df = pretty.melt(id_vars=["model"], value_vars=long_cols,
                              var_name="metric", value_name="value")
        try:
            import altair as alt

            chart = (
                alt.Chart(long_df)
                .mark_bar()
                .encode(
                    x=alt.X("model:N", sort="-y", title=None),
                    y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
                    color="metric:N",
                    column=alt.Column("metric:N", title=None),
                    tooltip=["model", "metric", "value"],
                )
                .properties(width=180, height=260)
            )
            st.altair_chart(chart, use_container_width=False)
        except Exception:
            st.bar_chart(pretty.set_index("model")[long_cols], height=320)

    # Confusion matrices
    if "confusion_matrix" in df.columns:
        st.markdown("### Confusion matrices")
        cm_cols = st.columns(min(3, len(df)))
        for i, row in df.iterrows():
            cm = row.get("confusion_matrix")
            if not cm or len(cm) != 4:
                continue
            tn, fp, fn, tp = cm
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["actual REAL", "actual FAKE"],
                columns=["pred REAL", "pred FAKE"],
            )
            with cm_cols[i % len(cm_cols)]:
                st.caption(row["model"])
                st.dataframe(cm_df, use_container_width=True)

    # Val-loss curves
    histories = _load_all_histories()
    if histories:
        st.markdown("### Validation-loss curves")
        loss_rows = []
        for name, hist in histories.items():
            for h in hist:
                if "val_loss" in h and "epoch" in h:
                    loss_rows.append({"model": name, "epoch": int(h["epoch"]),
                                      "val_loss": float(h["val_loss"])})
        if loss_rows:
            ldf = pd.DataFrame(loss_rows)
            pivot = ldf.pivot(index="epoch", columns="model", values="val_loss")
            st.line_chart(pivot, height=320)


# ----------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------

def _render_pyvis(ent_rows) -> None:
    from pyvis.network import Network

    net = Network(height="380px", width="100%", directed=True, bgcolor="#ffffff")
    net.add_node("ARTICLE", label="ARTICLE", color="#444444")
    for r in ent_rows:
        nid = r.get("qid") or r.get("mention")
        net.add_node(nid, label=r.get("mention"), title=r.get("description") or "")
        net.add_edge("ARTICLE", nid, label=r.get("type"))
    out = Path("artifacts/entity_graph.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))
    html = out.read_text(encoding="utf-8")
    import streamlit.components.v1 as components

    components.html(html, height=420, scrolling=True)


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------

def main() -> None:
    st.title("EKNet -- News Credibility Assessment (English)")
    st.caption(
        "Reimplementation of *An Entity Ontology-Based Knowledge Graph Embedding Approach "
        "to News Credibility Assessment* (Liu et al., 2024)."
    )

    tab_a, tab_b, tab_c = st.tabs([
        "Assess Article",
        "Compare Models",
        "Model Metrics",
    ])
    with tab_a:
        tab_assess()
    with tab_b:
        tab_compare()
    with tab_c:
        tab_metrics()


if __name__ == "__main__":
    main()
