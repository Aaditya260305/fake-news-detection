"""End-to-end training entry point.

Examples
--------
Train EKNet ablations (paper Table III):

    python -m src.train --model eknet --mode text_only
    python -m src.train --model eknet --mode ontology_only
    python -m src.train --model eknet --mode both

Train a baseline (paper Table I):

    python -m src.train --model textrnn
    python -m src.train --model textrcnn
    python -m src.train --model transformer_small
    python -m src.train --model fasttext
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config, ensure_dirs
from .data import (
    load_real_or_fake,
    load_fake_news_detection,
    stratified_split,
)
from .evaluation.metrics import classification_report
from .feature_cache import FeatureCache, clear_cache
from .features import FeatureSet
from .models.eknet import EKNet


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
#  utilities
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _section(title: str) -> None:
    log.info("=" * 70)
    log.info("=  %s", title)
    log.info("=" * 70)


def _prepare_features_eknet(cfg, splits, *, force_rebuild: bool = False):
    """Build text + kg features for the three splits.

    Honours the disk cache at ``artifacts/feature_cache/`` so repeat
    runs (e.g. the three EKNet ablation modes) skip the deterministic
    Step 1-5 work and just re-train the MLP head. Pass
    ``force_rebuild=True`` (CLI ``--rebuild-features``) to redo the
    full pipeline.
    """
    import time

    from .features import FeatureSet
    from .models.text_encoder import (
        GloveLoader,
        TextEncoder,
        MiniLMEncoder,
        build_text_encoder,
    )
    from .models.kg_encoder import KGEncoder
    from .kg.transE import KGEmbeddings, train_transE
    from .preprocessing.tfidf import fit_tfidf
    from .preprocessing.tokenize_en import tokenize_articles_batch, join_tokens
    from .entity_linking.kb_cache import KBCache
    from .entity_linking.wikidata_linker import WikidataLinker
    from .kg.build_article_kg import build_corpus

    train, val, test = splits
    all_articles = train + val + test
    log.info("Corpus sizes: train=%d  val=%d  test=%d  total=%d",
             len(train), len(val), len(test), len(all_articles))

    # ----------------------------------------------------------------------
    #  Try cache first (skip the whole pipeline if it matches)
    # ----------------------------------------------------------------------
    fcache = FeatureCache(cfg)
    log.info("Feature cache fingerprint: %s", fcache.digest)
    if force_rebuild:
        log.info("--rebuild-features set; ignoring any cached features.")
    elif fcache.is_valid():
        log.info("Cache HIT -- loading features from %s ...", fcache.paths.features)
        feats = fcache.load()
        log.info(
            "Loaded cached features (train=%d val=%d test=%d  text_dim=%d  kg_dim=%d)",
            feats["train"].text_emb.shape[0],
            feats["val"].text_emb.shape[0],
            feats["test"].text_emb.shape[0],
            feats["train"].text_emb.shape[1],
            feats["train"].kg_emb.shape[1],
        )
        return feats
    elif fcache.paths.fingerprint.exists():
        log.info(
            "Cache MISS -- config changed since last build. Differences:\n%s",
            fcache.explain_mismatch(),
        )
    else:
        log.info("No feature cache present -- building from scratch.")

    # ----------------------------------------------------------------------
    # 1) Tokenize the entire corpus ONCE (re-used for tf-idf + features).
    # ----------------------------------------------------------------------
    _section("Step 1/5  Tokenising entire corpus")
    t0 = time.time()
    all_tokens = tokenize_articles_batch(
        [a.text for a in all_articles],
        spacy_model=cfg.preprocessing.spacy_model,
        max_sentences=cfg.preprocessing.max_sentences,
        max_tokens_per_sentence=cfg.preprocessing.max_tokens_per_sentence,
        batch_size=64,
        desc="tokenise",
    )
    log.info("Tokenised %d articles in %.1fs (%.1f art/s)",
             len(all_tokens), time.time() - t0, len(all_tokens) / max(time.time() - t0, 1e-6))
    by_id: dict[str, list[list[str]]] = {
        a.id: toks for a, toks in zip(all_articles, all_tokens)
    }

    # ----------------------------------------------------------------------
    # 2) Fit tf-idf on the TRAIN split only.
    # ----------------------------------------------------------------------
    _section("Step 2/5  Fitting tf-idf on train split")
    t0 = time.time()
    train_joined = [join_tokens(by_id[a.id]) for a in train]
    tfidf = fit_tfidf(train_joined)
    log.info("tf-idf vocabulary size=%d (%.1fs)",
             len(tfidf.vectorizer.vocabulary_), time.time() - t0)

    enc_type = getattr(cfg.text_encoder, "type", "glove_tfidf")
    log.info("Building text encoder (type=%s)", enc_type)
    text_enc = build_text_encoder(cfg, tfidf=tfidf)
    log.info("Text encoder ready (dim=%d)", text_enc.dim)

    # ----------------------------------------------------------------------
    # 3) Build per-article knowledge graphs from cached Wikidata entries.
    # ----------------------------------------------------------------------
    _section("Step 3/5  Building per-article KGs (entity linking is cached)")
    t0 = time.time()
    kb_cache = KBCache(cfg.paths.kb_cache)
    crossview_path = Path(cfg.paths.kb_cache) / "crossview.json"
    crossview: dict[str, str] = {}
    if crossview_path.exists():
        with open(crossview_path, "r", encoding="utf-8") as f:
            crossview = json.load(f)
        log.info("Loaded crossview map with %d entities", len(crossview))
    else:
        log.warning("crossview.json missing -- run `python -m src.ontology.build_schema` first.")

    # OFFLINE-ONLY linker during training: cache-miss = "no entity"
    # rather than a live Wikidata call (which would slow each
    # iteration by rate_limit_s seconds). Use the warm script
    # (`python -m src.entity_linking.kb_cache --warm`) to populate the
    # cache before training instead.
    linker = WikidataLinker(
        kb_cache,
        endpoint=cfg.entity_linking.wikidata_endpoint,
        user_agent=cfg.entity_linking.wikidata_user_agent,
        top_relations=cfg.entity_linking.top_relations,
        offline_only=True,
    )

    per_article, global_triples = build_corpus(
        all_articles,
        linker=linker,
        crossview=crossview,
        ner_model=cfg.entity_linking.spacy_ner_model,
        keep_labels=cfg.ontology.top_level_types,
        max_mentions=cfg.entity_linking.max_entities_per_article,
    )
    if linker._offline_miss:
        log.info(
            "Linker offline-mode cache misses: %d "
            "(re-run `python -m src.entity_linking.kb_cache --warm --max-mentions 0` to cover more)",
            linker._offline_miss,
        )
    a2e = {kg.article_id: kg.entity_ids for kg in per_article}
    log.info("KG built: %d articles, %d total triples (%.1fs)",
             len(per_article), len(global_triples), time.time() - t0)

    # ----------------------------------------------------------------------
    # 4) Train TransE on the global KG.
    # ----------------------------------------------------------------------
    _section("Step 4/5  Training TransE (KG embeddings)")
    t0 = time.time()
    if global_triples:
        kg_emb = train_transE(
            global_triples,
            dim=cfg.kg.embedding_dim,
            epochs=cfg.kg.transe_epochs,
            batch_size=cfg.kg.transe_batch_size,
        )
        log.info("TransE done: %d entities, %d relations (%.1fs)",
                 len(kg_emb.entity_to_idx), len(kg_emb.relation_to_idx), time.time() - t0)
        from .kg.membership_loss import refine_with_membership

        if cfg.kg.membership_loss_weight > 0 and crossview:
            log.info("Applying OntoEA-style membership refinement (weight=%.2f) ...",
                     float(cfg.kg.membership_loss_weight))
            kg_emb.entity_emb = refine_with_membership(
                kg_emb.entity_emb,
                kg_emb.entity_to_idx,
                crossview,
                weight=float(cfg.kg.membership_loss_weight),
                iterations=5,
            )
    else:
        log.warning("No KG triples available; KG branch will be zero.")
        kg_emb = KGEmbeddings(
            entity_to_idx={},
            relation_to_idx={},
            entity_emb=np.zeros((1, cfg.kg.embedding_dim), dtype=np.float32),
            relation_emb=np.zeros((1, cfg.kg.embedding_dim), dtype=np.float32),
        )

    kg_enc = KGEncoder(kg_emb)

    # ----------------------------------------------------------------------
    # 5) Build per-article text + kg feature vectors.
    # ----------------------------------------------------------------------
    _section("Step 5/5  Encoding text + KG features")
    t0 = time.time()
    feats = {}
    use_minilm_batch = isinstance(text_enc, MiniLMEncoder)
    for split_name, articles in zip(("train", "val", "test"), (train, val, test)):
        log.info("Encoding split=%s (%d articles)", split_name, len(articles))
        ys = [int(a.label) for a in articles]
        ids = [a.id for a in articles]
        kg_vs = [kg_enc.encode(a2e.get(a.id, [])) for a in articles]

        if use_minilm_batch:
            # one big batch through the transformer = ~10x faster than per-article
            sents_per_art = [by_id.get(a.id, []) for a in articles]
            text_arr = text_enc.encode_articles_batch(sents_per_art, show_progress=True)
        else:
            from tqdm import tqdm as _tqdm
            text_vs = []
            for art in _tqdm(articles, desc=f"encode-{split_name}", unit="art"):
                text_vs.append(text_enc.encode_article(by_id.get(art.id, [])))
            text_arr = np.stack(text_vs, axis=0)

        feats[split_name] = FeatureSet(
            text_emb=text_arr,
            kg_emb=np.stack(kg_vs, axis=0),
            labels=np.asarray(ys, dtype=np.int64),
            ids=ids,
        )
    log.info("Features built (%.1fs). text_dim=%d  kg_dim=%d",
             time.time() - t0,
             feats["train"].text_emb.shape[1],
             feats["train"].kg_emb.shape[1])

    # ----------------------------------------------------------------------
    #  Save to cache so subsequent runs skip Steps 1-5.
    # ----------------------------------------------------------------------
    try:
        fcache.save(
            feats,
            tfidf=tfidf,
            kg_embeddings=kg_emb,
            article_to_entities=a2e,
            crossview=crossview,
        )
    except Exception as e:
        log.warning("Could not save feature cache: %s", e)
    return feats


# ----------------------------------------------------------------------
#  EKNet training loop
# ----------------------------------------------------------------------

def _train_eknet(cfg, mode: str, feats: dict[str, FeatureSet], outdir: Path) -> dict:
    train_fs = feats["train"]
    val_fs = feats["val"]
    test_fs = feats["test"]

    text_dim = train_fs.text_emb.shape[1]
    kg_dim = train_fs.kg_emb.shape[1]
    model = EKNet(
        text_dim=text_dim,
        kg_dim=kg_dim,
        mode=mode,
        hidden_dims=tuple(cfg.model.hidden_dims),
        dropout=cfg.model.dropout,
    )

    def _loader(fs: FeatureSet, shuffle: bool):
        ds = TensorDataset(
            torch.from_numpy(fs.text_emb).float(),
            torch.from_numpy(fs.kg_emb).float(),
            torch.from_numpy(fs.labels).float(),
        )
        return DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=shuffle)

    train_loader = _loader(train_fs, shuffle=True)
    val_loader = _loader(val_fs, shuffle=False)
    test_loader = _loader(test_fs, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.model.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        for tx, kg, y in train_loader:
            opt.zero_grad()
            logits = model(tx, kg)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach()) * y.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tx, kg, y in val_loader:
                val_loss += float(loss_fn(model(tx, kg), y)) * y.shape[0]
        val_loss /= len(val_loader.dataset)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        log.info("ep %02d  train=%.4f  val=%.4f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.training.early_stopping_patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- evaluate ----
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for tx, kg, y in test_loader:
            probs = torch.sigmoid(model(tx, kg))
            preds = (probs >= 0.5).long()
            y_true.extend(y.long().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    report = classification_report(y_true, y_pred, y_prob)

    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / f"eknet_{mode}.pt")
    with open(outdir / f"eknet_{mode}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"model": f"eknet_{mode}", **report}


# ----------------------------------------------------------------------
#  Baseline dispatcher
# ----------------------------------------------------------------------

def _train_baseline(cfg, name: str, splits, outdir: Path) -> dict:
    from .baselines.fasttext_baseline import train as train_fasttext
    from .baselines.textrnn import train as train_textrnn
    from .baselines.textrcnn import train as train_textrcnn
    from .baselines.transformer_small import train as train_transformer

    fn = {
        "fasttext": train_fasttext,
        "textrnn": train_textrnn,
        "textrcnn": train_textrcnn,
        "transformer_small": train_transformer,
    }[name]
    return fn(cfg, splits, outdir)


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--model", default="eknet", help="eknet | fasttext | textrnn | textrcnn | transformer_small")
    p.add_argument("--mode", default=None, help="text_only | ontology_only | both (eknet only)")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--rebuild-features",
        action="store_true",
        help="recompute the EKNet feature cache from scratch (ignore artifacts/feature_cache/)",
    )
    p.add_argument(
        "--clear-feature-cache",
        action="store_true",
        help="delete artifacts/feature_cache/ before running",
    )
    p.add_argument("--epochs", type=int, default=None, help="override baselines.epochs / training.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override baselines.batch_size / training.batch_size")
    p.add_argument("--max-len", type=int, default=None, help="override baselines.max_len")
    p.add_argument("--num-threads", type=int, default=None, help="override baselines.num_threads")
    p.add_argument(
        "--text-encoder",
        choices=["glove_tfidf", "minilm"],
        default=None,
        help="override text_encoder.type (minilm needs `pip install sentence-transformers`)",
    )
    p.add_argument(
        "--profile",
        choices=["fast", "balanced", "accuracy"],
        default=None,
        help=(
            "preset bundles that override several settings at once:\n"
            "  fast     -> small MLP, GloVe-100, few epochs (current defaults)\n"
            "  balanced -> deeper MLP, more KG epochs, augmentation on\n"
            "  accuracy -> MiniLM text encoder + deep MLP + long training"
        ),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    # Apply profile presets BEFORE the explicit --foo overrides below,
    # so an explicit flag can still override the profile.
    if args.profile == "balanced":
        cfg.model.hidden_dims = [256, 128, 64]
        cfg.model.dropout = 0.4
        cfg.training.epochs = 40
        cfg.training.early_stopping_patience = 6
        # Native PyTorch TransE on CPU is ~4x slower at dim=128 vs 64.
        # 40 epochs * dim 96 lands at ~6-8 min of TransE training on CPU
        # with early stopping usually triggering by epoch 25-30 anyway.
        cfg.kg.transe_epochs = 40
        cfg.kg.embedding_dim = 96
        cfg.entity_linking.max_entities_per_article = 15
        log.info("Profile=balanced: deeper MLP, moderate KG (CPU-friendly)")
    elif args.profile == "accuracy":
        cfg.text_encoder.type = "minilm"
        cfg.model.hidden_dims = [384, 192, 64]
        cfg.model.dropout = 0.4
        cfg.training.epochs = 60
        cfg.training.early_stopping_patience = 8
        # The text encoder is where 'accuracy' actually wins (MiniLM).
        # The KG branch is a small additive signal -- no need to crank
        # TransE epochs past the point of diminishing returns.
        cfg.kg.transe_epochs = 50
        cfg.kg.embedding_dim = 128
        cfg.entity_linking.max_entities_per_article = 20
        log.info("Profile=accuracy: MiniLM + deep MLP + long training")

    if args.text_encoder is not None:
        cfg.text_encoder.type = args.text_encoder
    if args.epochs is not None:
        cfg.baselines.epochs = args.epochs
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.baselines.batch_size = args.batch_size
        cfg.training.batch_size = args.batch_size
    if args.max_len is not None:
        cfg.baselines.max_len = args.max_len
    if args.num_threads is not None:
        cfg.baselines.num_threads = args.num_threads

    if args.clear_feature_cache:
        removed = clear_cache()
        log.info("Cleared feature cache (%d files removed).", removed)

    log.info("Loading primary dataset ...")
    items = load_real_or_fake(cfg)
    train, val, test = stratified_split(
        items,
        cfg.dataset.split.train,
        cfg.dataset.split.val,
        cfg.dataset.split.test,
        seed=cfg.seed,
    )
    log.info("train=%d  val=%d  test=%d", len(train), len(val), len(test))
    splits = (train, val, test)

    outdir = Path(args.out or "artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.model == "eknet":
        mode = args.mode or cfg.model.mode
        feats = _prepare_features_eknet(cfg, splits, force_rebuild=args.rebuild_features)
        report = _train_eknet(cfg, mode, feats, outdir)
    else:
        report = _train_baseline(cfg, args.model, splits, outdir)

    # write report
    rpath = outdir / f"{args.model}{('_' + args.mode) if args.mode else ''}_report.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("wrote %s", rpath)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
