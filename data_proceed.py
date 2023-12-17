import json
import os

def data_proceed(data_path, tgt_path):
    f = open(data_path,'r', encoding='utf-8',errors='ignore')
    data = json.load(f)
    f.close()
    new_data = []
    new_data_ = [] #这是asr改成manual的版本
    for sentences in data: #sentences 依然是一个列表
        new_sentences = []
        new_sentences_ = []
        for sentence in sentences:
            flag = True
            for semantic in sentence['semantic']: #依然是一个列表
                if not (semantic[2] in sentence['asr_1best']):
                    flag = False
                    break
            if flag:
                new_sentences.append(sentence)
                sentence['asr_1best'] = sentence['manual_transcript']
                new_sentences_.append(sentence)
        new_data.append(new_sentences)
        new_data_.append(new_sentences_)
    new_data += new_data_
    t = open(tgt_path, 'w',encoding='utf-8')
    json_str = json.dumps(new_data ,ensure_ascii=False,indent=4)
    t.write(json_str)
    t.close()
    return


if __name__ == "__main__":
    c = os.getcwd()
    data_path = os.path.join(c, "data/train.json")
    tgt_path = os.path.join(c, "data/train_.json")
    data_proceed(data_path, tgt_path)