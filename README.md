# NLU Joint model for chinese with word embedding
implementation base on: https://github.com/DSKSD/RNN-for-Joint-NLU

What changed:
1. Fix some bug in source.
2. Add word embedding support, test passed on 64 size word2vec.
3. Add predict function, already tested on test set.

Description:
1. model.py -> NLP model with encoder and decoder.
2. preprocess.py -> preprocessing raw data. including word segment, stop words removing.
3. data.py -> utility class used to load training / test data from batch.
4. train.py -> train data.
5. service.py -> websocket service.
6. main.py -> entrypoint for training / test / configuration

Things todo:
1. Replace LSTM with GRU.
2. Add F1 score and confusion matrix for accuracy.
