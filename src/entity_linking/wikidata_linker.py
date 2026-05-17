"""Mention -> Wikidata QID linker.

Uses the official MediaWiki ``wbsearchentities`` endpoint, then fetches
each entity's claims via ``wbgetentities`` to extract a small set of
configured relations (``P31`` instance_of, ``P279`` subclass_of, ...).

Production-grade rate-limit hygiene
-----------------------------------
Wikidata's terms of use ask third-party scripts to:

* Send a descriptive User-Agent that contains contact info.
* Stay within roughly 1 req/s sustained (their global default).
* Honour the ``Retry-After`` header on HTTP 429.
* Avoid hammering after a 5xx.

Concretely, this module:

* Retries on 429 / 5xx with exponential back-off + jitter.
* Reads ``Retry-After`` and obeys it.
* Sleeps between successful requests (default 1.0 s).
* Never caches a *failure*: a 429 leaves the entry uncached so a
  later resume will retry it.
* Batches ``wbgetentities`` (the MediaWiki API allows up to 50 IDs
  in a single call).

All successful results are written through ``KBCache`` so subsequent
runs are fully offline.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import requests

from .kb_cache import KBCache


log = logging.getLogger(__name__)


# Sentinel value cached when we positively know a mention has *no*
# Wikidata match (a real "empty" response from the API). Distinct from
# "we haven't tried yet" (= absent from the cache) and from "we tried
# but got 429" (= absent from the cache, intentionally).
_NEGATIVE_SEARCH: dict = {"_no_match": True}


@dataclass
class LinkedEntity:
    mention: str
    qid: str | None
    label: str | None
    description: str | None
    claims: dict[str, list[str]] = field(default_factory=dict)
    score: float = 0.0

    def is_resolved(self) -> bool:
        return self.qid is not None


class WikidataLinker:
    """Mention->QID resolver with cache + retry/back-off."""

    def __init__(
        self,
        cache: KBCache,
        endpoint: str = "https://www.wikidata.org/w/api.php",
        user_agent: str = "EKNet-Reimpl/1.0 (https://github.com/; educational)",
        top_relations: Iterable[str] = ("P31", "P279", "P17", "P569", "P27"),
        rate_limit_s: float = 1.0,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        timeout_s: float = 15.0,
        offline_only: bool = False,
    ):
        """Mention -> Wikidata resolver.

        Parameters
        ----------
        offline_only
            If True, the linker will NEVER make a live HTTP request.
            Cache-miss returns ``LinkedEntity(qid=None, ...)`` immediately.
            This is the mode you want during training / feature
            building -- live API calls there would silently slow each
            iteration by ``rate_limit_s`` seconds per uncached mention.
            Use the explicit ``--warm`` script (with offline_only=False)
            to populate the cache up front.
        """
        self.cache = cache
        self.endpoint = endpoint
        self.user_agent = user_agent
        self.top_relations = tuple(top_relations)
        self.rate_limit_s = float(rate_limit_s)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.timeout_s = float(timeout_s)
        self.offline_only = bool(offline_only)
        self._offline_miss = 0

        # request-side counters (reset by ``reset_stats``).
        self._cache_hits = 0          # served from local JSONL
        self._live_resolved = 0       # successful Wikidata fetches
        self._live_unresolved = 0     # Wikidata fetched but found no QID
        self._live_failed = 0         # transient errors (429, 5xx, network)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            }
        )
        self._last_call = 0.0

    # ------------------------------------------------------------------
    #  HTTP plumbing
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        wait = self.rate_limit_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _request(self, params: dict, *, what: str) -> dict | None:
        """Send a GET, returning the parsed JSON dict or None on failure.

        Returning None means "transient failure -- caller should NOT
        cache anything so a future run can retry".
        """
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            try:
                r = self._session.get(self.endpoint, params=params, timeout=self.timeout_s)
            except requests.exceptions.RequestException as e:
                log.warning("Wikidata %s connection error (%s) attempt=%d", what, e, attempt)
                if attempt > self.max_retries:
                    return None
                self._sleep_backoff(attempt)
                continue

            # 429 / 503 -- retry with respect for Retry-After.
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After")
                wait = self._compute_wait(attempt, ra)
                log.warning(
                    "Wikidata %s %d %s -> sleeping %.1fs (attempt=%d/%d)",
                    what, r.status_code, "Too Many Requests" if r.status_code == 429 else "Service Unavailable",
                    wait, attempt, self.max_retries,
                )
                if attempt > self.max_retries:
                    return None
                time.sleep(wait)
                continue
            # other 5xx -- short backoff
            if 500 <= r.status_code < 600:
                if attempt > self.max_retries:
                    log.warning("Wikidata %s %d (final) -- giving up.", what, r.status_code)
                    return None
                self._sleep_backoff(attempt)
                continue
            # other 4xx -- the request itself is bad, do NOT retry.
            if 400 <= r.status_code < 500:
                log.warning("Wikidata %s %d for %s -- not retrying.", what, r.status_code, params)
                return {}

            try:
                return r.json()
            except ValueError as e:
                log.warning("Wikidata %s non-JSON response (%s)", what, e)
                if attempt > self.max_retries:
                    return None
                self._sleep_backoff(attempt)

    def _sleep_backoff(self, attempt: int) -> None:
        wait = min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_max_s)
        wait += random.uniform(0, 0.5)
        time.sleep(wait)

    def _compute_wait(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
        return min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_max_s)

    # ------------------------------------------------------------------
    #  observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return per-request counters since construction or the last
        ``reset_stats``. Useful for the Streamlit demo to show users
        whether an article's entities came from the local cache or were
        fetched live from Wikidata."""
        return {
            "offline_only": self.offline_only,
            "cache_hits": self._cache_hits,
            "live_resolved": self._live_resolved,
            "live_unresolved": self._live_unresolved,
            "live_failed": self._live_failed,
            "offline_miss": self._offline_miss,
        }

    def reset_stats(self) -> None:
        self._cache_hits = 0
        self._live_resolved = 0
        self._live_unresolved = 0
        self._live_failed = 0
        self._offline_miss = 0

    # ------------------------------------------------------------------
    #  search
    # ------------------------------------------------------------------

    def search(self, mention: str, *, ner_label: str | None = None) -> LinkedEntity:
        """Resolve a mention to a Wikidata entity.

        If ``ner_label`` is given (e.g. ``"PERSON"``, ``"ORG"``, ``"GPE"``),
        we ask Wikidata for the top-5 candidates and pick the one whose
        description keywords match the NER tag. This dramatically
        reduces wrong-sense links like ``Pentagon -> Brussels district``
        or ``Gloucester City -> football club``.

        Cache entries from NER-aware lookups live under the key
        ``"{NER_LABEL}|{mention}"`` so they do not collide with the
        legacy mention-only cache from training.
        """
        m = _clean_mention(mention)
        if not m:
            return LinkedEntity(mention=mention, qid=None, label=None, description=None)

        cache_key = f"{ner_label}|{m}" if ner_label else m

        # 1. NER-qualified cache hit
        cached = self.cache.get_search(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return _from_cache_search(mention, cached)

        # 2. Bare (legacy) cache hit -- accept only if its description
        #    is consistent with the NER label (or no NER label given).
        if ner_label:
            bare = self.cache.get_search(m)
            if bare is not None and _description_matches_ner(bare, ner_label):
                self._cache_hits += 1
                return _from_cache_search(mention, bare)
        else:
            bare = self.cache.get_search(m)
            if bare is not None:
                self._cache_hits += 1
                return _from_cache_search(mention, bare)

        # 3. Offline mode -- accept a bare cache hit even if mismatched
        #    (better than zero signal), otherwise give up silently.
        if self.offline_only:
            if ner_label:
                bare = self.cache.get_search(m)
                if bare is not None:
                    self._cache_hits += 1
                    return _from_cache_search(mention, bare)
            self._offline_miss += 1
            return LinkedEntity(mention=mention, qid=None, label=None, description=None)

        # 4. Live search -- pull 5 candidates, pick the best NER match
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": m,
            "limit": 10 if ner_label else 1,
            "type": "item",
        }
        data = self._request(params, what="search")
        if data is None:
            self._live_failed += 1
            return LinkedEntity(mention=mention, qid=None, label=None, description=None)

        results = data.get("search", []) if isinstance(data, dict) else []
        if not results:
            self.cache.put_search(cache_key, _NEGATIVE_SEARCH)
            self._live_unresolved += 1
            return LinkedEntity(mention=mention, qid=None, label=None, description=None)

        best = _best_candidate(results, ner_label)
        record = {
            "qid": best.get("id"),
            "label": best.get("label"),
            "description": best.get("description"),
            "score": float(best.get("match", {}).get("score", 0) or 0),
        }
        self.cache.put_search(cache_key, record)
        self._live_resolved += 1
        return _from_cache_search(mention, record)

    # ------------------------------------------------------------------
    #  claims
    # ------------------------------------------------------------------

    def fetch_claims(self, qid: str) -> dict[str, list[str]]:
        if not qid:
            return {}
        cached = self.cache.get_claims(qid)
        if cached is not None:
            return _filter_claims(cached, self.top_relations)

        if self.offline_only:
            self._offline_miss += 1
            return {}

        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "claims|labels|descriptions",
            "languages": "en",
            "format": "json",
        }
        data = self._request(params, what="claims")
        if data is None:
            return {}

        flat = _extract_top_claims(data.get("entities", {}).get(qid, {}))
        self.cache.put_claims(qid, flat)
        return _filter_claims(flat, self.top_relations)

    def fetch_claims_batch(self, qids: Iterable[str]) -> dict[str, dict[str, list[str]]]:
        """Fetch claims for up to 50 QIDs in a single API call.

        Cached QIDs are filtered out before the request.
        """
        wanted: list[str] = []
        out: dict[str, dict[str, list[str]]] = {}
        for q in qids:
            if not q:
                continue
            cached = self.cache.get_claims(q)
            if cached is not None:
                out[q] = _filter_claims(cached, self.top_relations)
            else:
                wanted.append(q)

        if self.offline_only:
            self._offline_miss += len(wanted)
            return out

        for batch_start in range(0, len(wanted), 50):
            batch = wanted[batch_start : batch_start + 50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "claims|labels|descriptions",
                "languages": "en",
                "format": "json",
            }
            data = self._request(params, what="claims-batch")
            if data is None:
                # transient failure -- skip this batch; retry on next run
                continue
            for q in batch:
                flat = _extract_top_claims(data.get("entities", {}).get(q, {}))
                self.cache.put_claims(q, flat)
                out[q] = _filter_claims(flat, self.top_relations)
        return out

    # ------------------------------------------------------------------
    #  combined entry point
    # ------------------------------------------------------------------

    def link(self, mention: str, *, ner_label: str | None = None) -> LinkedEntity:
        ent = self.search(mention, ner_label=ner_label)
        if ent.qid:
            ent.claims = self.fetch_claims(ent.qid)
        return ent

    def link_many(self, mentions: Iterable[str]) -> list[LinkedEntity]:
        return [self.link(m) for m in mentions]


# ----------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------

_BAD_LEAD_CHARS = "\"'`(){}[]<>,.;:!?@#$%^&*+=|/\\~ \t\r\n"

# Leading English articles / honorifics that spaCy NER tends to keep
# inside the entity span. wbsearchentities is whitespace-sensitive and
# will fail to match e.g. ``the Strait of Hormuz`` against the
# canonical Wikidata label ``Strait of Hormuz``. Strip these greedily.
_LEAD_FILLERS = (
    "the ", "a ", "an ",
    "mr. ", "mrs. ", "ms. ", "dr. ", "prof. ", "sir ", "lady ",
)


def _clean_mention(s: str) -> str:
    """Strip stray punctuation, whitespace, and leading English fillers.

    NER taggers sometimes hand us substrings like ``\"Force Protection``
    or ``Bravo:`` or ``the Strait of Hormuz``. Trimming such cruft
    saves a *lot* of pointless API calls and lets wbsearchentities
    actually find the entity.
    """
    if not s:
        return ""
    out = s.strip()
    # repeatedly trim leading/trailing junk
    while out and out[0] in _BAD_LEAD_CHARS:
        out = out[1:]
    while out and out[-1] in _BAD_LEAD_CHARS:
        out = out[:-1]
    # strip leading English filler words ("the Strait of Hormuz" -> "Strait of Hormuz")
    lower = out.lower()
    for filler in _LEAD_FILLERS:
        if lower.startswith(filler):
            out = out[len(filler):]
            break
    return out


# ---- NER-label aware candidate ranking ---------------------------------
#
# wbsearchentities returns the top-scoring candidates by string overlap,
# with no notion of article context. For a news pipeline we usually
# know the spaCy NER label of the mention -- which is a strong signal
# about *which sense* of an ambiguous string is meant. We exploit that
# by asking for top-5 candidates and picking the highest-ranked one
# whose Wikidata description contains a keyword consistent with the
# NER tag.
#
# This is intentionally cheap (no extra HTTP calls -- we just inspect
# the descriptions that wbsearchentities already returns) and works
# remarkably well: ``Pentagon`` (NER=ORG) skips Q18435290 ("inner city
# of Brussels") because its description matches GPE, not ORG, and
# falls through to the US Department of War / Pentagon building, which
# matches "building" / "headquarters".

_NER_DESC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "PERSON": (
        "person", "human", "politician", "actor", "actress", "singer",
        "musician", "writer", "author", "poet", "artist", "athlete",
        "scientist", "physicist", "chemist", "biologist", "mathematician",
        "philosopher", "engineer", "businessman", "businessperson",
        "businesswoman", "journalist", "soldier", "general", "officer",
        "president", "senator", "monarch", "king", "queen", "prince",
        "princess", "founding father", "designer", "upholsterer",
        "seamstress", "minister", "diplomat", "lawyer", "judge", "doctor",
        "physician", "nurse", "teacher", "professor", "director",
        "producer", "composer", "painter", "sculptor",
    ),
    "ORG": (
        "organization", "organisation", "company", "corporation",
        "agency", "institute", "institution", "association",
        "society", "party", "department", "ministry", "club",
        "non-profit", "nonprofit", "team", "council", "committee",
        "school", "university", "college", "consortium", "alliance",
        "headquarters", "building", "office", "bureau", "service",
        "force", "army", "navy", "air force", "coalition", "union",
        "federation",
    ),
    "GPE": (
        "country", "city", "town", "village", "state", "province",
        "capital", "republic", "kingdom", "municipality", "county",
        "district", "borough", "settlement", "metropolis", "commonwealth",
        "territory", "nation", "sovereign",
    ),
    "LOC": (
        "mountain", "river", "lake", "ocean", "sea", "island",
        "continent", "forest", "desert", "valley", "peninsula",
        "bay", "gulf", "strait", "region", "area", "plateau",
        "archipelago", "reef",
    ),
    "FAC": (
        "building", "structure", "airport", "stadium", "bridge",
        "tunnel", "monument", "museum", "library", "hospital",
        "facility", "tower", "skyscraper", "fort", "fortress",
        "headquarters", "complex", "plaza",
    ),
    "EVENT": (
        "war", "battle", "event", "summit", "competition",
        "conference", "festival", "election", "olympics", "championship",
        "tournament", "ceremony", "anniversary", "revolution",
    ),
    "PRODUCT": (
        "product", "vehicle", "device", "model", "software",
        "weapon", "aircraft", "ship", "car", "smartphone",
    ),
    "WORK_OF_ART": (
        "film", "movie", "novel", "song", "album", "book",
        "painting", "play", "tv series", "drama", "sculpture",
        "documentary",
    ),
    "LAW": (
        "law", "act", "amendment", "treaty", "constitution",
        "convention", "statute", "code", "regulation",
    ),
}


def _description_matches_ner(record: Mapping[str, object] | None,
                             ner_label: str | None) -> bool:
    """True if the Wikidata ``description`` text plausibly belongs to
    the NER label. Used to validate a cached entry before we trust it
    in NER-aware mode."""
    if not record or record.get("_no_match"):
        return False
    if not ner_label or ner_label not in _NER_DESC_KEYWORDS:
        return True
    desc = str(record.get("description") or "").lower()
    if not desc:
        return False
    return any(kw in desc for kw in _NER_DESC_KEYWORDS[ner_label])


def _best_candidate(results: list[dict], ner_label: str | None) -> dict:
    """Pick the top-scoring candidate whose description matches the
    NER label's expected types. Fall back to ``results[0]`` if none
    match (rare for popular entities, common for novel/obscure ones)."""
    if not results:
        return {}
    if not ner_label or ner_label not in _NER_DESC_KEYWORDS:
        return results[0]
    keywords = _NER_DESC_KEYWORDS[ner_label]
    for r in results:
        desc = str(r.get("description") or "").lower()
        if any(kw in desc for kw in keywords):
            return r
    return results[0]


def _from_cache_search(mention: str, record: Mapping[str, object]) -> LinkedEntity:
    if not record or record.get("_no_match"):
        return LinkedEntity(mention=mention, qid=None, label=None, description=None)
    return LinkedEntity(
        mention=mention,
        qid=record.get("qid"),                    # type: ignore[arg-type]
        label=record.get("label"),                # type: ignore[arg-type]
        description=record.get("description"),    # type: ignore[arg-type]
        score=float(record.get("score", 0) or 0), # type: ignore[arg-type]
    )


def _extract_top_claims(entity_payload: dict) -> dict[str, list[str]]:
    flat: dict[str, list[str]] = {}
    for prop, values in (entity_payload.get("claims", {}) or {}).items():
        ids: list[str] = []
        for v in values:
            try:
                val = v["mainsnak"]["datavalue"]["value"]
            except Exception:
                continue
            if isinstance(val, dict) and "id" in val:
                ids.append(val["id"])
            elif isinstance(val, str):
                ids.append(val)
            elif isinstance(val, dict) and "time" in val:
                ids.append(val["time"])
        if ids:
            flat[prop] = ids[:5]
    return flat


def _filter_claims(claims: dict[str, list[str]], keep: tuple[str, ...]) -> dict[str, list[str]]:
    return {p: v for p, v in claims.items() if p in keep}
