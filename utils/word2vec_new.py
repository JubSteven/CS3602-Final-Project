import hanlp
word2vec = hanlp.load(hanlp.pretrained.word2vec.CONVSEG_W2V_NEWS_TENSITE_WORD_PKU)
print(word2vec('先进'))

print(word2vec("帮忙"))

import jieba

# 你的中文句子
sentence = "北京天安门上太阳升"

# 使用jieba进行分词
words = jieba.cut(sentence,cut_all=True,HMM=True)

# 将分词结果转换为列表
words_list = list(words)

print(words_list)
