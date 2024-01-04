import jieba
import nltk
from nltk.corpus import wordnet
import random


# 如果还没有下载，需要下载nltk相关的数据
# nltk.download('omw-1.4')
# nltk.download('wordnet')

def find_synonym(word):
    synonyms = set()
    for syn in wordnet.synsets(word, lang='cmn'):  # 使用中文语言码
        for lemma in syn.lemma_names('cmn'):
            synonyms.add(lemma)
    return list(synonyms - {word})


def replace_in_semantic(semantic, original_word, new_word):
    for item in semantic:
        item_words = list(jieba.cut(item[2]))
        if original_word in item_words:
            # 替换同义词
            item[2] = item[2].replace(original_word, new_word)
    return semantic


def random_delete(words, probability):
    if random.random() < probability:
        if len(words) > 1:  # 防止列表为空
            del words[random.randint(0, len(words) - 1)]
    return words


def random_shuffle(words, probability):
    if random.random() < probability and len(words) > 1:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return words


def replace_with_synonym(text, semantic, synonym_prob=0.2, delete_prob=0.2, shuffle_prob=0.15):
    # 使用 jieba 进行分词
    words = list(jieba.cut(text))

    # 同义词替换
    for i, word in enumerate(words):
        if random.random() < synonym_prob:
            synonyms = find_synonym(word)
            if synonyms:
                original_word = word
                new_word = random.choice(synonyms)
                words[i] = new_word

                # 更新 semantic 字段
                semantic = replace_in_semantic(semantic, original_word, new_word)

    # 随机删除
    words = random_delete(words, delete_prob)

    # 随机交换
    words = random_shuffle(words, shuffle_prob)

    return ''.join(words), semantic


# sample = {
#     "utt_id": 1,
#     "manual_transcript": "导航到凯里大十字",
#     "asr_1best": "导航到凯里大十字",
#     "semantic": [
#         ["inform", "操作", "导航"],
#         ["inform", "终点名称", "凯里大十字"]
#     ]
# }
#
# new_asr_1best, new_semantic = replace_with_synonym(sample["asr_1best"], sample["semantic"])
# print("New asr_1best:", new_asr_1best)
# print("Updated Semantic:", new_semantic)

if __name__ == "__main__":
    import json
    from tqdm import tqdm

    train_data = json.load(open("data/train.json", "r"))
    augmented_data = []
    for _, sample_grp in tqdm(enumerate(train_data)):
        augmented_data_grp = []
        for sample in sample_grp:
            new_asr_1best, new_semantic = replace_with_synonym(sample["asr_1best"],
                                                               sample["semantic"],
                                                               synonym_prob=0.2,
                                                               delete_prob=0.1,
                                                               shuffle_prob=0.2
                                                               )
            sample["asr_1best"] = new_asr_1best
            sample["semantic"] = new_semantic
            augmented_data_grp.append(sample)
        augmented_data.append(augmented_data_grp)

    # concatenate raw train data and augmented data
    train_data.extend(augmented_data)

    # save augmented data
    json.dump(train_data, open("data/train_augmented.json", "w"), indent=4, ensure_ascii=False)
