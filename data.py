import torch
import pickle
import random
import os
from torch.autograd import Variable
from gensim.models import KeyedVectors

USE_CUDA = torch.cuda.is_available()


class DataLoader(object):

    def __init__(self, data_path, sequence_max_length, batch_size):
        self.data_path = data_path
        self.sequence_max_length = sequence_max_length
        self.batch_size = batch_size

    def load_train(self):
        processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        print("processed_data_path : %s" % processed_path)

        if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
            train_data, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path, "processed_train_data.pkl"), "rb"))
            return train_data, word2index, tag2index, intent2index

        if not os.path.exists(processed_path):
            os.makedirs(processed_path)

        data = open(self.data_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(data))
        data = [t.rstrip('\n') for t in data]  # remove \n in train
        data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]]
                for t in data]
        data = [[t[0][1:-1], t[1], t[2]] for t in data]
        seq_in, seq_out, intent = list(zip(*data))

        flatten = lambda l: [item for sublist in l for item in sublist]

        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}".format(
            vocab=len(vocab), slot_tag=len(slot_tag), intent_tag=len(intent_tag)))

        word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for token in vocab:
            if token not in word2index.keys():
                word2index[token] = len(word2index)

        tag2index = {'<PAD>': 0}
        for tag in slot_tag:
            if tag not in tag2index.keys():
                tag2index[tag] = len(tag2index)

        intent2index = {}
        for ii in intent_tag:
            if ii not in intent2index.keys():
                intent2index[ii] = len(intent2index)

        processed_data = self.process_data(word2index, tag2index, intent2index, data)

        pickle.dump((processed_data, word2index, tag2index, intent2index),
                    open(os.path.join(processed_path, "processed_train_data.pkl"), "wb"))
        pickle
        print("Preprocessing data complete!")

        return processed_data, word2index, tag2index, intent2index

    def process_data(self, word2index, tag2index, intent2index, data=None):

        word2vec_model = KeyedVectors.load_word2vec_format('./data/word2vec.bin', binary=True)

        if data is None:
            data = open(self.data_path, "r").readlines()
            print("Successfully load data. # of set : %d " % len(data))
            data = [t.rstrip('\n') for t in data]  # remove \n in train
            data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]]
                    for t in data]
            data = [[t[0][1:-1], t[1], t[2]] for t in data]

        seq_in, seq_out, intent = list(zip(*data))

        sin = []
        sout = []

        for i in range(len(seq_in)):
            temp = seq_in[i]
            if len(temp) < self.sequence_max_length:
                temp.append('<EOS>')
                while len(temp) < self.sequence_max_length:
                    temp.append('<PAD>')
            else:
                temp = temp[:self.sequence_max_length]
                temp[-1] = '<EOS>'
            sin.append(temp)

            temp = seq_out[i]
            if len(temp) < self.sequence_max_length:
                while len(temp) < self.sequence_max_length:
                    temp.append('<PAD>')
            else:
                temp = temp[:self.sequence_max_length]
                temp[-1] = '<EOS>'
            sout.append(temp)

        data = list(zip(sin, sout, intent))

        processed_data = []
        for item in data:
            temp = self.prepare_sequence(item[0], word2index)
            temp = temp.view(1, -1)

            temp2 = self.prepare_sequence(item[1], tag2index)
            temp2 = temp2.view(1, -1)

            temp3 = Variable(torch.LongTensor([intent2index[item[2]]])).cuda() if USE_CUDA else Variable(
                torch.LongTensor([intent2index[item[2]]]))

            temp4 = self.embedding_sentence(item[0], word2vec_model)
            temp4 = temp4.view(1, self.sequence_max_length, -1)

            # word2index, embedding sentence, tag2index, intent2index,
            processed_data.append((temp, temp4, temp2, temp3))

        return processed_data

    def load_test(self):
        processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        print("processed_data_path : %s" % processed_path)

        _, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path, "processed_train_data.pkl"), "rb"))

        processed_data = self.process_data(word2index, tag2index, intent2index)

        print("Preprocessing test complete!")
        return processed_data, word2index, tag2index, intent2index

    def prepare_sequence(self, sequence, to_ix):
        idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], sequence))
        tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
        return tensor

    def get_batch(self, data):
        random.shuffle(data)
        data_size = len(data)
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, data_size)
            yield data[start_idx: end_idx]

    def embedding_sentence(self, sentence, word2vec_model):
        embedding_unknown = [0 for i in range(word2vec_model.vector_size)]
        embedding_sentence = []
        for word in sentence:
            if word in word2vec_model.wv.vocab:
                embedding_sentence.append(word2vec_model[word])
            else:
                embedding_sentence.append(embedding_unknown)

        tensor = Variable(torch.FloatTensor(embedding_sentence)).cuda() if USE_CUDA else Variable(torch.FloatTensor(embedding_sentence))
        return tensor
