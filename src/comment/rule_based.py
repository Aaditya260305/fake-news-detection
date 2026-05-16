"""Rule + KG-mismatch driven comment generator (Sec. IV-B analogue).

The paper trains a BiLSTM+LSTM Encoder-Decoder with attention, PGN, and
Coverage to *learn* short comments. Because the public English datasets
do not ship reference comments, we instead produce comments
algorithmically from signals that are *meaningful* for credibility:

* ``ner_label`` vs. ``Wikidata instance_of`` mismatches (e.g., article
  asserts a name as a PERSON, but Wikidata says it is a Q_book).
* Mentions that fail to resolve to a QID at all (suspicious names).
* The top-K FastTextRank keywords (gives a "what is this about" angle).

We still expose a ROUGE evaluation against article titles (proxy
reference) so the comment-generation component is measurable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..entity_linking.kb_cache import KBCache
from ..entity_linking.ner_spacy import extract_mentions
from ..entity_linking.wikidata_linker import LinkedEntity, WikidataLinker
from ..preprocessing.fasttextrank import textrank_keywords


log = logging.getLogger(__name__)


@dataclass
class CommentResult:
    text: str
    flagged_entities: list[str]
    flagged_reasons: list[str]
    keywords: list[str]


# spaCy NER label -> a small set of Wikidata superclasses we expect
EXPECTED_SUPERCLASSES: dict[str, set[str]] = {
    "PERSON": {"Q5"},                       # human
    "ORG": {"Q43229", "Q4830453", "Q783794"},  # organization / business / company
    "GPE": {"Q6256", "Q515", "Q3624078"},      # country / city / sovereign state
    "LOC": {"Q17334923", "Q35145263"},      # location / geographic feature
    "EVENT": {"Q1656682"},                  # event
    "WORK_OF_ART": {"Q838948"},             # work of art
    "PRODUCT": {"Q2424752"},                # product
}


def _superclass_mismatch(ner_label: str, instance_of_qids: list[str]) -> bool:
    expected = EXPECTED_SUPERCLASSES.get(ner_label)
    if not expected or not instance_of_qids:
        return False
    return not (expected & set(instance_of_qids))


def generate_comment(
    article_text: str,
    *,
    linker: WikidataLinker,
    ner_model: str = "en_core_web_sm",
    keep_labels=None,
    top_keywords: int = 3,
    max_flags: int = 5,
) -> CommentResult:
    mentions = extract_mentions(
        article_text, model=ner_model, keep_labels=keep_labels, max_mentions=25
    )
    flagged: list[str] = []
    reasons: list[str] = []
    for m in mentions:
        ent = linker.link(m.text)
        if not ent.qid:
            flagged.append(m.text)
            reasons.append(f"{m.text!r} could not be found in Wikidata")
            continue
        instance_of = (ent.claims or {}).get("P31", [])
        if _superclass_mismatch(m.label, instance_of):
            flagged.append(m.text)
            reasons.append(
                f"{m.text!r} is tagged as {m.label} but Wikidata classifies {ent.label or ent.qid} differently"
            )
        if len(flagged) >= max_flags:
            break

    kws = [w for w, _ in textrank_keywords(article_text, top_k=top_keywords)]

    if flagged:
        sent = (
            f"This article references {', '.join(flagged)} but the linked Wikidata records "
            f"do not match the claims in context."
        )
    else:
        sent = "All extracted entities resolve cleanly against Wikidata."
    sent_b = f"Key topics: {', '.join(kws)}." if kws else ""
    return CommentResult(
        text=(sent + " " + sent_b).strip(),
        flagged_entities=flagged,
        flagged_reasons=reasons,
        keywords=kws,
    )


def evaluate_with_rouge(predictions: list[str], references: list[str]) -> dict:
    from ..evaluation.metrics import rouge_scores

    return rouge_scores(predictions, references)
