"""Rule-based KG-mismatch comment generator (paper Sec. IV-B analogue).

What it does
------------
The original paper trains a seq2seq model (BiLSTM encoder + LSTM
decoder + attention + Pointer-Generator Network + Coverage) on
**Chinese Weibo rumor-comment pairs** to learn short credibility
comments. No public English dataset ships those ground-truth comments,
so a *learned* decoder cannot be trained or evaluated fairly here.

Instead we produce comments **algorithmically** from three KG-grounded
signals that are themselves meaningful for credibility -- so the output
is still a useful piece of explanatory text, just not a learned one:

1. **NER label vs. Wikidata `instance_of` mismatch.** If spaCy tags
   ``"Hogwarts"`` as ``PERSON`` but Wikidata classifies that QID under
   ``Q838948 (work of art)``, the entity is flagged. This is the
   "ontology-class disagreement" we already use as an EKNet feature,
   reused here as a credibility-explanation signal.
2. **Unresolvable mentions.** If a named entity from the text has no
   Wikidata QID at all, it's flagged as suspicious (fabricated names
   are a classic fake-news tell).
3. **Top-K FastTextRank keywords.** Give the comment a one-line
   "what is this article about" angle so the output isn't just a list
   of red flags.

The two-sentence template the generator emits looks like:

    "This article references {flagged entities} but the linked Wikidata
     records do not match the claims in context. Key topics: {keywords}."

When nothing is flagged the first sentence becomes:

    "All extracted entities resolve cleanly against Wikidata."

This is **not** a learned model. Its purpose is twofold:

* In the **Streamlit demo** it provides a human-readable explanation
  to accompany the EKNet verdict.
* In **paper Table II** it gives us *some* ROUGE row we can honestly
  reproduce in English. ROUGE is computed against the article title
  as a proxy reference, configured by
  ``cfg.comment_generator.rouge_reference_field`` in
  ``configs/default.yaml``.

For the actual paper Table II row of "EKNet+PGN+CVG" you would need to
swap ``generate_comment`` with a learned seq2seq trained on
(article, ground-truth-comment) pairs. That is out of scope for this
CPU-friendly reimplementation; the gap is discussed honestly in
``reports/table2_assessment.md``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..entity_linking.kb_cache import KBCache
from ..entity_linking.ner_spacy import extract_mentions
from ..entity_linking.wikidata_linker import (
    LinkedEntity,
    WikidataLinker,
    _description_matches_ner,
)
from ..preprocessing.fasttextrank import textrank_keywords


log = logging.getLogger(__name__)


@dataclass
class CommentResult:
    text: str
    flagged_entities: list[str]
    flagged_reasons: list[str]
    keywords: list[str]


# Hard-coded P31 superclass hints. Kept as a *secondary* check: the
# primary signal is the entity's English description (matched via the
# linker's NER-keyword table), which is far more flexible. This set
# only catches the few common cases where Wikidata stores a useful
# direct ``instance_of`` but the description text is missing/sparse.
EXPECTED_SUPERCLASSES: dict[str, set[str]] = {
    "PERSON": {"Q5", "Q15632617"},                          # human / fictional human
    "ORG": {                                                # any kind of organisation
        "Q43229", "Q4830453", "Q783794", "Q22865",          # generic + US federal dept
        "Q327333", "Q1530705", "Q2659904", "Q163740",       # gov agency / non-profit
        "Q484652", "Q748019", "Q7210356", "Q11691",         # int'l org / regulator / party / exchange
        "Q11032", "Q15265344", "Q3918", "Q9826",            # newspaper / broadcaster / university
        "Q1664720", "Q31855", "Q294163", "Q15911314",       # institute / association
        "Q7188",                                            # government
    },
    "GPE": {
        "Q6256", "Q515", "Q3624078", "Q3957",               # country / city / state / town
        "Q56061", "Q1093829", "Q35657", "Q1549591",         # admin entity / US city / US state
        "Q5119", "Q486972", "Q852446", "Q1352230",          # capital / settlement / US township
    },
    "LOC": {"Q17334923", "Q35145263"},
    "EVENT": {"Q1656682", "Q198", "Q178561"},               # event / war / battle
    "WORK_OF_ART": {"Q838948", "Q11424", "Q7725634"},       # work / film / literary work
    "PRODUCT": {"Q2424752"},
}


def _superclass_mismatch(
    ner_label: str,
    instance_of_qids: list[str],
    *,
    record: dict | None = None,
) -> bool:
    """Return True iff the entity's type clearly disagrees with the NER label.

    Two-stage check:
      1. Description-keyword match (primary, works for any P31).
      2. Hard-coded P31 superclass set (fallback, for entities whose
         description is empty or unhelpful).

    Returns True only when *both* signals fail to confirm the type, so
    correctly-linked entities with unusual P31 (e.g. ``Q22865`` for the
    DoD) are not falsely flagged.
    """
    if record is not None and _description_matches_ner(record, ner_label):
        return False
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
    # Only check real named-entity types -- skip numeric/quantitative
    # spaCy labels (DATE, TIME, MONEY, PERCENT, ORDINAL, CARDINAL,
    # QUANTITY) that Wikidata search cannot meaningfully resolve and
    # that would otherwise pollute the flag list with noise like
    # "'28 feet' could not be found in Wikidata".
    if keep_labels is None:
        keep_labels = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}

    mentions = extract_mentions(
        article_text, model=ner_model, keep_labels=keep_labels, max_mentions=25
    )
    flagged: list[str] = []
    reasons: list[str] = []
    for m in mentions:
        ent = linker.link(m.text, ner_label=m.label)
        if not ent.qid:
            flagged.append(m.text)
            reasons.append(f"{m.text!r}: no Wikidata match found")
            continue
        instance_of = (ent.claims or {}).get("P31", [])
        record = {"qid": ent.qid, "label": ent.label, "description": ent.description}
        if _superclass_mismatch(m.label, instance_of, record=record):
            flagged.append(m.text)
            desc = ent.description or "(no description)"
            reasons.append(
                f"{m.text!r} (NER: {m.label}) -> linked to "
                f"'{ent.label or ent.qid}' [{desc}], which does not "
                f"look like a {m.label}"
            )
        if len(flagged) >= max_flags:
            break

    kws = [w for w, _ in textrank_keywords(article_text, top_k=top_keywords)]

    if flagged:
        sent = (
            f"Potential entity-linking concerns for: {', '.join(flagged)}. "
            f"These are heuristic observations from a string-based Wikidata "
            f"lookup and may simply reflect a wrong link rather than a "
            f"factual problem with the article."
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
