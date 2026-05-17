### Table II -- Fake News Assessment (Comment Generation)

| Method                        | ROUGE-1 | ROUGE-2 | ROUGE-L | source    |
|-------------------------------|---------|---------|---------|-----------|
| RNN [paper]                   |  15.55  |   2.83  |  13.08  | paper     |
| RNN-context [paper]           |  15.85  |   3.23  |  14.48  | paper     |
| SummaReranker [paper]         |  23.24  |   1.32  |  22.72  | paper     |
| EKNet [paper]                 |  17.58  |   3.43  |  15.34  | paper     |
| EKNet+PGN [paper]             |  24.87  |   3.85  |  23.42  | paper     |
| EKNet+PGN+CVG [paper]         |  27.45  |   4.33  |  25.65  | paper     |
| Rule-based KG-mismatch (ours) |  12.30  |   2.45  |   9.90  | this repo |

#### What each row means

- **RNN**: Vanilla recurrent encoder-decoder. The article body is fed to a BiLSTM encoder; an LSTM decoder emits the comment one token at a time. No copy mechanism, no entity awareness. This is the weakest of the paper's baselines.
- **RNN-context**: Same as RNN but the decoder also conditions on a small context window around each entity mention. Helps the decoder stay on topic but still cannot copy rare names verbatim.
- **SummaReranker**: Strong extractive-summarisation baseline (Liu & Lapata 2022 style). Generates several candidate comments and re-ranks them with a learned classifier. Wins on ROUGE-1/L but is content-agnostic (no KG signal).
- **EKNet**: The paper's EKNet credibility model with a plain seq2seq comment head: text + KG embeddings -> LSTM decoder. No copy mechanism.
- **EKNet+PGN**: EKNet with a **Pointer-Generator Network**: at each decoding step the model can either generate from the vocabulary or *copy* a token directly from the article (great for proper nouns and rare entities).
- **EKNet+PGN+CVG**: EKNet + PGN + **Coverage Mechanism**: an extra attention-coverage loss discourages the decoder from repeating the same source tokens. This is the paper's best system.
- **Rule-based KG-mismatch (ours)**: Our **non-trainable** comment generator. For each article it (a) runs spaCy NER, (b) links every mention to Wikidata, (c) flags entities whose NER label disagrees with the Wikidata `instance_of` claim (e.g. tagged as PERSON but Wikidata says `Q838948 work of art`), and (d) extracts the top FastTextRank keywords. The output is a two-sentence template:
    "This article references X, Y, Z but the linked Wikidata records do not match the claims in context. Key topics: A, B, C."
ROUGE is computed against the article title as a proxy reference. This row is **fully reproduced locally** -- no learned decoder.

#### Example comments produced by our generator

- *Article (id=0, true label=FAKE):* "You Can Smell Hillary’s Fear"
  *Generated comment:* This article references Shillman Journalism Fellow, the Freedom Center, New York, Islam, FBI but the linked Wikidata records do not match the claims in context. Key topics: the, a, of.

- *Article (id=3978, true label=FAKE):* "IOWA FARMER CLAIMS BILL CLINTON HAD SEX WITH COW DURING ‘COCAINE PARTY’"
  *Generated comment:* This article references Sioux Falls, IA, 1992, Tom Brady’s, Tom Brady, Willow Brady Jr., Clinton but the linked Wikidata records do not match the claims in context. Key topics: the, i, and.

- *Article (id=1582, true label=REAL):* "The math is with Hillary: She’s surging in the polls — and many Republicans are in denial"
  *Generated comment:* This article references Trump, 19 percent, Clinton, seven, the Electoral College but the linked Wikidata records do not match the claims in context. Key topics: the, is, to.



*Rows tagged `[paper]` are reproduced verbatim from the original paper (Liu et al., 2024, Table II) for context; they are not recomputed locally because the paper trains a learned seq2seq + PGN + Coverage decoder on Chinese rumor-comment pairs that no public English dataset ships. Our row is computed on the **Real or Fake** test split (n=200) with the article title used as the proxy reference, as configured in `configs/default.yaml -> comment_generator.rouge_reference_field`. All ROUGE values are F1 percentages.*
