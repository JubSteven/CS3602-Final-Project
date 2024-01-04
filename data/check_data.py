import json

data = json.load(open("/Users/husky/Three-Year-Aut/CS3602-NLP_/nlp-dev/wrong_examples/try_2023-12-16-21-51-38/epoch=9_batch=100_wrong_examples.json", "r", encoding="utf-8"))
print(len(data))
cnt = 0
align = 0

sentence = []
for item in data:
    item_list = data[item]
    for utt in item_list:
        sentence.append(utt["sentence"])

print(len(sentence))
dev = json.load(open("/Users/husky/Three-Year-Aut/CS3602-NLP_/nlp-dev/data/development.json", "r", encoding="utf-8"))

cnt = 0
for item in dev:
    for utt in item:
        if utt["asr_1best"] in sentence:
            cnt += 1
            if utt["manual_transcript"] == utt["asr_1best"]:
                align += 1

print(cnt)

print(align / cnt)




