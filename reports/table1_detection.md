### Table I -- Fake News Detection (Real or Fake dataset)

| Method       | Precision | Recall | F1    | Miss-rate |
|--------------|-----------|--------|-------|-----------|
| FastText     | 0.942     | 0.924  | 0.933 | 0.076     |
| TextRNN      | 1.000     | 0.145  | 0.253 | 0.855     |
| TextRCNN     | 0.718     | 0.861  | 0.783 | 0.139     |
| Transformer  | 0.890     | 0.457  | 0.604 | 0.543     |
| EKNet (ours) | 0.882     | 0.845  | 0.863 | 0.155     |