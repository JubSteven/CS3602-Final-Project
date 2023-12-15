import jieba
import logging
from text2vec import SentenceModel

sentences = ["你好世界",
             "美国人民",
             "如果一个字在一句话中出现了不止一次",
             "我原本以为两个字就是两个向量拼一起"]
    

def char_word_pair(batch_sentences):
    jieba.setLogLevel(logging.ERROR)
    cut_words = []
    for sentence in batch_sentences:
        # result = jieba.tokenize(sentence, mode="default")
        cut_word = list(jieba.cut(sentence, cut_all=True))
        cut_words.append(cut_word)
    
    batch_pair_list = []
    for i, sentence_cut_words in enumerate(cut_words):
        pair_list = [set() for _ in range(len(batch_sentences[i]))]
        for each_word in sentence_cut_words:
            for character in each_word:
                indices = [i for i, targ in enumerate(batch_sentences[i]) if targ == character]
                
                for idx in indices:
                    pair_list[idx].add(each_word)
                    
        batch_pair_list.append(pair_list)
    
    # Transform back to set and record W.
    W = 0
    for pair_list in batch_pair_list:
        for i in range(len(pair_list)):
            pair_list[i] = list(pair_list[i])
            W = max(len(pair_list[i]), W)
    
    # Pad all the paired lists to W
    for pair_list in batch_pair_list:
        for i in range(len(pair_list)):
            pair_list[i].extend(["" for _ in range(W - len(pair_list[i]))])
    
    
def word_embed(char_word_pair):
    model = SentenceModel()
    for pair_list in char_word_pair:
        for pair in pair_list:
            for i in range(len(pair)):
                pair[i] = model.encode(pair[i])
            
    print(char_word_pair.shape)
              
pairs = char_word_pair(sentences)
word_embed(pairs)