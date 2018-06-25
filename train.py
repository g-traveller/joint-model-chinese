import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from data import *
from model import Encoder,Decoder

USE_CUDA = torch.cuda.is_available()


def train(config):

    data_loader = DataLoader(config.file_path, config.max_length, config.batch_size)
    train_data, word2index, tag2index, intent2index = data_loader.load_train()
    
    if train_data is None:
        print("Please check your data or its path")
        return
    
    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), config.hidden_size * 2)
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.init_weights()
    decoder.init_weights()

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    
    for step in range(config.step_size):
        losses = []
        for i, batch in enumerate(data_loader.get_batch(train_data)):
            x, embedding_x, y_1, y_2 = zip(*batch)
            x = torch.cat(x)
            embedding_x = torch.cat(embedding_x)
            tag_target = torch.cat(y_1)
            intent_target = torch.cat(y_2)
            x_mask = torch.cat([Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(len(batch), -1)

            encoder.zero_grad()
            decoder.zero_grad()

            output, hidden_c = encoder(x, embedding_x, x_mask)
            start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * len(batch)])).cuda().transpose(1, 0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<SOS>']] * len(batch)])).transpose(1, 0)

            tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

            loss_1 = loss_function_1(tag_score, tag_target.view(-1))
            loss_2 = loss_function_2(intent_score, intent_target)

            loss = loss_1 + loss_2
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy())
            loss.backward()

            torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()

            if i % 100 == 0:
                print("Step", step, " : ", np.mean(losses))
                losses = []
    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    torch.save(encoder, os.path.join(config.model_dir, 'jointnlu-encoder.pt'))
    torch.save(decoder, os.path.join(config.model_dir, 'jointnlu-decoder.pt'))
    print("Training Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/loreal.train.iob' ,
                        help='path of train data')
    parser.add_argument('--embedding_path', type=str, default='./data/word2vec.bin' ,
                        help='path of word embedding')
    parser.add_argument('--model_dir', type=str, default='./models/' ,
                        help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60,
                        help='max sequence length')
    parser.add_argument('--embedding_size', type=int , default=64,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=64,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1,
                        help='number of layers in lstm')
    
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()
    train(config)
