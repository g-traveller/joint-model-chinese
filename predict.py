import argparse
from data import *


def predict(config):

    data_loader = DataLoader(config.file_path, config.max_length, config.batch_size)
    test_data, word2index, tag2index, intent2index = data_loader.load_test()

    if test_data is None:
        print("Please check your test data or its path")
        return

    # load model
    decoder = torch.load(os.path.join(config.model_dir, 'jointnlu-decoder.pt'))
    encoder = torch.load(os.path.join(config.model_dir, 'jointnlu-encoder.pt'))

    intent_correct_predict = 0
    tag_correct_predict = 0
    total_tags = 0

    for i, batch in enumerate(data_loader.get_batch(test_data)):
        x, embedding_x, y_1, y_2 = zip(*batch)
        x = torch.cat(x)
        embedding_x = torch.cat(embedding_x)
        tag_target = torch.cat(y_1)
        intent_target = torch.cat(y_2)
        x_mask = torch.cat([Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(len(batch), -1)

        encoder.zero_grad()
        decoder.zero_grad()

        output, hidden_c = encoder(x, embedding_x, x_mask)
        start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * len(batch)])).cuda().transpose(1, 0) if USE_CUDA else Variable(
            torch.LongTensor([[word2index['<SOS>']] * len(batch)])).transpose(1, 0)

        tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)
        # batch, sequence_max_length, tag_length
        tag_score = tag_score.view(len(batch), -1, list(tag_score[1].size())[0])

        # calculate intent detection accuracy
        _, max_index = intent_score.max(1)
        intent_correct_predict += (max_index == intent_target).sum()

        # calculate tag detection accuracy
        _, max_tag_index = tag_score.max(2)
        batch_sequence_length = torch.sum(x_mask.data == 0, 1)
        for idx, sequence_length in enumerate(batch_sequence_length):
            actual_tags = tag_target[idx][:sequence_length - 1]     # remove last <EOS> position
            predict_tags = max_tag_index[idx][:sequence_length - 1]

            tag_correct_predict += torch.sum(torch.eq(actual_tags, predict_tags) == 1)
            total_tags += (sequence_length - 1)

    print('Intent Detection Result: Total samples: {}, Predict correctly: {}, Accuracy: {}'
          .format(len(test_data), intent_correct_predict, round(float(intent_correct_predict) / len(test_data), 3)))

    print('Tags Detection Result: Total tags: {}, Predict correctly: {}, Accuracy: {}'
          .format(total_tags, tag_correct_predict, round(float(tag_correct_predict) / float(total_tags), 3)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/loreal.test.iob',
                        help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60,
                        help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=16)
    config = parser.parse_args()

    predict(config)
