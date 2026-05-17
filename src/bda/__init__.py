"""Big-data analytics layer for the EKNet pipeline.

The paper's title (Liu et al., 2024) advertises a *Big Data-Driven*
framework. This module makes the BDA story explicit by applying
streaming / sub-linear / probabilistic algorithms to our corpus, all
on CPU and all in pure Python so the techniques port unchanged to a
larger cluster setting.

Submodules
----------
sketches
    Count-Min Sketch and Bloom Filter -- sub-linear-memory
    aggregation primitives.

dedup
    MinHash + LSH near-duplicate detection. Standard "shingle the
    document, hash the shingles, hash the hashes" pipeline.

corpus_analytics
    Driver that loads the dataset, runs all of the above, and writes
    a self-contained Markdown report at
    ``reports/bda_corpus_stats.md``.

These pieces are intentionally framework-free (no PySpark, no Dask).
They mirror what those frameworks do under the hood and stay laptop-
friendly. Each module documents the equivalent
Spark / MapReduce / Flink primitive in its docstring.
"""
