# coding: utf-8

import jieba
import re
import asyncio
import websockets
import json
import torch.nn.functional as F
import math
from data import *
from gensim.models import KeyedVectors


class Classification(object):
    def __init__(self, model_name, word2vec_model, jieba, stop_words):
        self.jieba = jieba
        self.stop_words = stop_words
        model_dir = 'models'
        self.data_loader = DataLoader(None, 20, 64, model_dir, model_name, word2vec_model)
        self.decoder = torch.load(os.path.join(model_dir, model_name, 'jointnlu-decoder.pt'))
        self.encoder = torch.load(os.path.join(model_dir, model_name, 'jointnlu-encoder.pt'))

    def predict(self, literal):
        literal = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", literal)
        words = jieba.lcut(literal)
        data = [w for w in words if w not in stop_words]
        word_len = len(data)
        seq, embedding_seq, word2index, tag2index, intent2index = self.data_loader.process_one(data)
        seq_mask = torch.cat([Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in seq]).view(1, -1)

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        output, hidden_c = self.encoder(seq, embedding_seq, seq_mask)
        start_decode = Variable(torch.LongTensor([[word2index['<SOS>']]])).cuda().transpose(1, 0) if USE_CUDA else Variable(
            torch.LongTensor([[word2index['<SOS>']]])).transpose(1, 0)

        tag_score, intent_score = self.decoder(start_decode, hidden_c, output, seq_mask)
        _, max_index = intent_score.max(1)

        print(intent_score)
        print(F.softmax(intent_score))
        print(F.log_softmax(intent_score))
        max_intent_confidence = intent_score.squeeze()[max_index].data.tolist()[0]

        index2tag = {v: k for k, v in tag2index.items()}
        index2intent = {v: k for k, v in intent2index.items()}

        _, max_tag_index = tag_score.max(1)
        max_tag_confidence = (math.e ** (tag_score)).max(1) # get softmax
        tags = list(map(lambda ii: index2tag[ii], max_tag_index.data.tolist()))
        tag_list = []

        i = 0
        while i < word_len:
            if tags[i].startswith("B-"):
                tag = tags[i][2:]
                value = data[i]
                total_confidence = max_tag_confidence[0][i]
                count = 1.0
                j = i + 1
                while j < len(tags):
                    if tags[j] == "I-" + tag:
                        value += data[j]
                        j = j + 1
                        total_confidence += max_tag_confidence[0][j]
                        count = count + 1
                    else:
                        break
                tag_list.append({'name': tag, 'value': value, 'confidence': (total_confidence / float(count)).data.tolist()})

            i = i + 1

        result = [{
            'intent': index2intent[max_index.data.tolist()[0]],
            'confidence': max_intent_confidence,
            'slots': tag_list
        }]
        return result


async def socket(websocket, path):
    in_message = await websocket.recv()
    request = json.loads(in_message)
    model_name = request['botName']
    literal = request['message']

    classifier = model_dict[model_name]
    result = classifier.predict(literal)
    response = json.dumps(result, ensure_ascii=False)
    await websocket.send(response)


if __name__ == '__main__':
    word2vec_model = KeyedVectors.load('./data/word2vec.64.model')
    jieba.load_userdict('./data/dict.txt')
    with open('data/chinese_stopwords.txt', "r", encoding="utf8") as f:
        stop_words = [line.strip() for line in f.readlines()]

    model_list = ['CUHKChatbot']
    model_dict = {}
    for model in model_list:
        classification = Classification(model, word2vec_model, jieba, stop_words)
        model_dict[model] = classification

    # Start WebSocket
    socket_server = websockets.serve(socket, None, 8868)
    print("Prepare starting websockets server.")
    asyncio.get_event_loop().run_until_complete(socket_server)
    print("Websocket server started!")
    asyncio.get_event_loop().run_forever()
