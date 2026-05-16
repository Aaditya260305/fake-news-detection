"""EoBSchema -- the 4-dimensional hierarchical entity ontology.

Mirrors Fig. 2 of the paper:

    D1 (top-level type)
       D2 (subclass)
          D3 (leaf class)
             D4 (attribute group)

Each refined parameter is addressable by its dotted path, e.g.
``Person.Politician.HeadOfState.dob``.

The schema is *content-agnostic*: an analyst (or a script) populates it
with the dimensions appropriate to their corpus. We ship a default
populator (``build_schema.populate_from_wikidata``) that walks
``P31``/``P279`` chains for every QID in the KB cache and assembles a
sensible 4-level skeleton.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class OntologyNode:
    name: str
    level: int                 # 1..4
    parent: str | None = None
    children: dict[str, "OntologyNode"] = field(default_factory=dict)
    attributes: list[str] = field(default_factory=list)

    def add_child(self, name: str) -> "OntologyNode":
        if name not in self.children:
            self.children[name] = OntologyNode(name=name, level=self.level + 1, parent=self.name)
        return self.children[name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "parent": self.parent,
            "attributes": list(self.attributes),
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


class EoBSchema:
    """The 4-dimensional schema described in Sec. III-B."""

    NUM_DIMENSIONS = 4

    def __init__(self, top_level_types: Iterable[str] | None = None):
        self.root = OntologyNode(name="<ROOT>", level=0, parent=None)
        if top_level_types:
            for t in top_level_types:
                self.root.add_child(t)

    # ---- programmatic edits ----

    def add_path(self, path: list[str], attributes: list[str] | None = None) -> OntologyNode:
        """Add a dotted path; missing intermediate nodes are created.

        ``path`` must have length <= 4 (one entry per dimension).
        """
        assert 1 <= len(path) <= self.NUM_DIMENSIONS
        node = self.root
        for name in path:
            node = node.add_child(name)
        if attributes:
            for a in attributes:
                if a not in node.attributes:
                    node.attributes.append(a)
        return node

    def find(self, dotted: str) -> OntologyNode | None:
        node = self.root
        for part in dotted.split("."):
            if part not in node.children:
                return None
            node = node.children[part]
        return node

    def all_classes(self) -> list[str]:
        """Return dotted paths of every (D1..D3) class."""
        out: list[str] = []

        def _walk(n: OntologyNode, prefix: list[str]):
            if 1 <= n.level <= 3:
                out.append(".".join(prefix))
            for c in n.children.values():
                _walk(c, prefix + [c.name])

        for child in self.root.children.values():
            _walk(child, [child.name])
        return out

    # ---- serialisation ----

    def to_dict(self) -> dict:
        return self.root.to_dict()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "EoBSchema":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = cls()
        s.root = _node_from_dict(data)
        return s


def _node_from_dict(d: dict) -> OntologyNode:
    n = OntologyNode(name=d["name"], level=d["level"], parent=d.get("parent"))
    n.attributes = list(d.get("attributes", []))
    for k, v in d.get("children", {}).items():
        n.children[k] = _node_from_dict(v)
    return n
