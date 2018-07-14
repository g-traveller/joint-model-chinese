# NLU Joint model for chinese with word embedding
implementation base on: https://github.com/DSKSD/RNN-for-Joint-NLU

What changed:
1. Fix some bug in source.
2. Add word embedding support, test passed on 64 size word2vec.
3. Add predict function, already tested on test set.

Things todo:
1. Replace LSTM with GRU.
2. Add F1 score and confusion matrix for accuracy.
