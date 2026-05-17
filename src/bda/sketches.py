"""Sub-linear-memory data-structure primitives.

These are the textbook "big-data" sketches you'd use to mine a stream
that doesn't fit in RAM. They are useful here for two reasons:

1. They demonstrate the BDA techniques the paper alludes to in its
   title.
2. They let us answer corpus-wide questions ("which 50 entities are
   mentioned most often across 6k articles?", "is QID Q1234567 in the
   KG?") without ever building a full in-memory dictionary -- so the
   same code scales to 6 million articles unchanged.

We implement them in pure Python with the hash-mixing scheme used by
`Cassandra <https://cassandra.apache.org/doc/latest/cassandra/operating/bloom_filter.html>`_
and the original Count-Min Sketch paper (Cormode & Muthukrishnan,
2005).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


# ----------------------------------------------------------------------
#  hashing helpers
# ----------------------------------------------------------------------

_MERSENNE_PRIME = (1 << 61) - 1


def _h(seed: int, item: str) -> int:
    """Stable 61-bit hash. Combines Python's ``hash`` with a seed so
    several independent hash functions can be derived from one input."""
    return (hash((seed, item)) ^ (seed * 0x9E3779B97F4A7C15)) & _MERSENNE_PRIME


# ----------------------------------------------------------------------
#  Count-Min Sketch (Cormode & Muthukrishnan, 2005)
# ----------------------------------------------------------------------

@dataclass
class CountMinSketch:
    """Approximate frequency counter for streaming data.

    Equivalent to Spark's ``DataFrame.stat.countMinSketch(...)`` or
    Flink's ``CountAggregateFunction``. With ``depth`` rows and
    ``width`` columns the over-estimate on any item is bounded by
    ``2 * N / width`` with probability ``1 - (1/2)**depth``.

    Memory: ``depth * width`` ints, *independent of the number of
    distinct items.*

    Example
    -------
    >>> cms = CountMinSketch(width=2048, depth=5)
    >>> for tok in stream_of_billions_of_tokens:
    ...     cms.add(tok)
    >>> cms.estimate("president")
    412377
    """

    width: int = 2048
    depth: int = 5
    table: list[list[int]] = field(init=False)
    n: int = 0

    def __post_init__(self) -> None:
        self.table = [[0] * self.width for _ in range(self.depth)]

    def add(self, item: str, count: int = 1) -> None:
        for d in range(self.depth):
            self.table[d][_h(d + 1, item) % self.width] += count
        self.n += count

    def update_many(self, items: Iterable[str]) -> None:
        for it in items:
            self.add(it)

    def estimate(self, item: str) -> int:
        return min(self.table[d][_h(d + 1, item) % self.width] for d in range(self.depth))

    def heavy_hitters(self, candidates: Iterable[str], top_k: int = 50) -> list[tuple[str, int]]:
        """Return the ``top_k`` items from ``candidates`` ranked by
        their sketch estimate.

        Note: a Count-Min Sketch cannot enumerate items by itself
        (only query them). To find heavy hitters at true streaming
        scale you pair it with a Misra-Gries-style cache or a top-K
        heap; here we just feed back the unique candidates we already
        saw, which is fine since this code runs on a single machine.
        """
        seen: set[str] = set(candidates)
        scored = [(c, self.estimate(c)) for c in seen]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ----------------------------------------------------------------------
#  Bloom filter (Bloom, 1970)
# ----------------------------------------------------------------------

@dataclass
class BloomFilter:
    """Probabilistic set-membership test.

    Equivalent to Cassandra's row-key Bloom filter or Spark's broadcast
    join filter. ``contains`` may have false positives at rate
    ``fp_rate``, but ``add(x); assert x in bf`` is always true.

    Memory: ``ceil(-n * ln(fp) / (ln 2)^2)`` bits, *independent of the
    items' size*.
    """

    capacity: int = 100_000
    fp_rate: float = 0.01

    def __post_init__(self) -> None:
        m = -self.capacity * math.log(self.fp_rate) / (math.log(2) ** 2)
        k = (m / self.capacity) * math.log(2)
        self._m = max(int(math.ceil(m)), 8)
        self._k = max(int(round(k)), 1)
        self._bits = bytearray((self._m + 7) // 8)
        self._n = 0

    @property
    def num_bits(self) -> int:
        return self._m

    @property
    def num_hashes(self) -> int:
        return self._k

    def _positions(self, item: str) -> Iterable[int]:
        for s in range(self._k):
            yield _h(s + 1, item) % self._m

    def add(self, item: str) -> None:
        for p in self._positions(item):
            self._bits[p >> 3] |= 1 << (p & 7)
        self._n += 1

    def update_many(self, items: Iterable[str]) -> None:
        for it in items:
            self.add(it)

    def __contains__(self, item: str) -> bool:
        for p in self._positions(item):
            if not (self._bits[p >> 3] & (1 << (p & 7))):
                return False
        return True

    def stats(self) -> dict[str, float]:
        return {
            "capacity": self.capacity,
            "configured_fp_rate": self.fp_rate,
            "num_bits": float(self._m),
            "num_hashes": float(self._k),
            "memory_kb": self._m / 8 / 1024,
            "n_inserted": float(self._n),
        }
