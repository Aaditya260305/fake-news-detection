"""EoBData -- the triple store ``{(e_i, p_j, v_k)}`` of Sec. III-B.

Each triple is an edge in the entity-relation graph. We keep two indices
for fast lookup:

* ``by_entity``    -- e_i  -> list[(p_j, v_k)]
* ``by_class``     -- D3 class path -> list[entity_id]
* ``crossview``    -- entity_id -> ontology class path
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Triple:
    e: str          # entity id (e.g. Wikidata QID)
    p: str          # relation (Wikidata property)
    v: str          # value (QID or literal)


@dataclass
class EoBData:
    triples: list[Triple] = field(default_factory=list)
    by_entity: dict[str, list[Triple]] = field(default_factory=lambda: defaultdict(list))
    by_class: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    crossview: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    # ---- mutation ----

    def add_triple(self, t: Triple) -> None:
        self.triples.append(t)
        self.by_entity[t.e].append(t)

    def link_class(self, entity_id: str, class_path: str) -> None:
        self.crossview[entity_id] = class_path
        self.by_class[class_path].append(entity_id)

    def set_label(self, entity_id: str, label: str) -> None:
        self.labels[entity_id] = label

    # ---- queries ----

    def entities(self) -> list[str]:
        return list(self.by_entity.keys())

    def relation_pairs(self, entity_id: str) -> list[tuple[str, str]]:
        return [(t.p, t.v) for t in self.by_entity.get(entity_id, [])]

    def class_of(self, entity_id: str) -> str | None:
        return self.crossview.get(entity_id)

    def entities_of_class(self, class_path: str) -> list[str]:
        return list(self.by_class.get(class_path, []))

    # ---- serialisation ----

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "triples": [t.__dict__ for t in self.triples],
            "crossview": dict(self.crossview),
            "labels": dict(self.labels),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "EoBData":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        d = cls()
        for t in data.get("triples", []):
            d.add_triple(Triple(**t))
        for e, c in data.get("crossview", {}).items():
            d.link_class(e, c)
        d.labels.update(data.get("labels", {}))
        return d

    # ---- bulk construct ----

    @classmethod
    def from_linked_entities(
        cls,
        linked: Iterable,
        crossview: dict[str, str] | None = None,
    ) -> "EoBData":
        d = cls()
        for ent in linked:
            if not getattr(ent, "qid", None):
                continue
            if ent.label:
                d.set_label(ent.qid, ent.label)
            for p, vals in (ent.claims or {}).items():
                for v in vals:
                    d.add_triple(Triple(e=ent.qid, p=p, v=str(v)))
            if crossview and ent.qid in crossview:
                d.link_class(ent.qid, crossview[ent.qid])
        return d
