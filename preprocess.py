import jieba
import re


def process_text_file(read_file_path, write_file_path):

    jieba.load_userdict('data/user_dict.txt')

    with open('data/chinese_stopwords.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]

    with open(read_file_path, 'r') as r:
        with open(write_file_path, 'w') as w:
            for line in r:
                line, intent = line.rstrip('\n').rsplit(' ', 1)
                line = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", line)
                words = jieba.lcut(line)
                words = [w for w in words if w not in stop_words]
                line = 'BOS ' + ' '.join(words) + ' EOS'
                line += '\t' + len(words) * 'O ' # slot default filling
                line += intent # intent
                w.write(line + '\n')


if __name__ == '__main__':
    process_text_file('../data/loreal.train.raw.iob', '../data/loreal.train.iob')
