from train import train
from predict import predict
from gensim.models import KeyedVectors


class Config(object):
    def __init__(self, model_name, file_name, step):
        self.file_name = file_name
        self.model_dir = './models/'
        self.max_length = 40
        self.embedding_size = 64
        self.hidden_size = 64
        self.num_layers = 1
        self.step_size = step
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.model_name = model_name


def build_config(model_name, file_name, step):
    return Config(model_name, file_name, step)


if __name__ == '__main__':
    
    #word2vec_model = KeyedVectors.load_word2vec_format('./data/word2vec.bin', binary=True)
    #train(build_config('Loreal', 'loreal.train.iob', 500), word2vec_model)
    #predict(build_config('loreal', 'loreal.test.iob', None), word2vec_model)
    
    word2vec_model = KeyedVectors.load('data/word2vec.64.model')
    #word2vec_model = KeyedVectors.load('data/word2vec.300.model')

    #train(build_config('CUHKChatbot', 'CUHKChatbot_train.iob', 300), word2vec_model)
    predict(build_config('CUHKChatbot', 'CUHKChatbot_test.iob', None), word2vec_model)
