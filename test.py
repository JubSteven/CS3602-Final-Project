import jieba
import logging
from text2vec import SentenceModel
import os
import torch
import time

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

sentences = ["你好世界", "美国人民", "如果一个字在一句话中出现了不止一次", "我原本以为两个字就是两个向量拼一起"]


# >>>>>>@pengxiang added on 12.15
def char_word_pair(batch_sentences, pad_length=None):
    """
        transform a batch of Chinese sentence into a list of char-word pairs
        Example: "美国人民" -> ["美国", "美国", "人民", "人民"]
        
    Args:
        batch_sentences (list): A list of (unpadded) Chinese sentences
    """
    jieba.setLogLevel(logging.ERROR)

    if pad_length is None:
        pad_length = max([len(each) for each in batch_sentences])

    cut_words = []
    for sentence in batch_sentences:
        # result = jieba.tokenize(sentence, mode="default")
        cut_word = list(jieba.cut(sentence, cut_all=True))
        cut_words.append(cut_word)

    batch_pair_list = []
    for i, sentence_cut_words in enumerate(cut_words):
        pair_list = [set() for _ in range(len(batch_sentences[i]))]  # use set to prevent redundance.
        for each_word in sentence_cut_words:
            for character in each_word:
                indices = [i for i, targ in enumerate(batch_sentences[i]) if targ == character]

                for idx in indices:
                    pair_list[idx].add(each_word)

        batch_pair_list.append(pair_list)

    # Transform set back to list and record W.
    W = 0
    for pair_list in batch_pair_list:
        for i in range(len(pair_list)):
            pair_list[i] = list(pair_list[i])
            W = max(len(pair_list[i]), W)

    # Pad the word length of each sentence to pad_length
    for i in range(len(batch_pair_list)):
        batch_pair_list[i].extend([[""] for _ in range(pad_length - len(batch_pair_list[i]))])

    # Pad all the paired lists of each char in each sentence to W
    for pair_list in batch_pair_list:
        for i in range(len(pair_list)):
            pair_list[i].extend(["" for _ in range(W - len(pair_list[i]))])

    return batch_pair_list


def word_embed(char_word_pair):
    """Takes in original char_word_pairs and transform them into vectors using text2vec sentence model.

    Args:
        char_word_pair (list [list [list]]): [[["美国"], ["美国"], ["人民"], ["人民"], [""]] ..]
    """

    model = SentenceModel()
    for pair_list in char_word_pair:
        for pair in pair_list:
            for i in range(len(pair)):
                pair[i] = model.encode(pair[i])

    embedded_char_word_pair = torch.Tensor(char_word_pair)
    return embedded_char_word_pair


start = time.time()
pairs = char_word_pair(sentences)
print(time.time() - start)
word_embed(pairs)
end = time.time()

print(end - start)
