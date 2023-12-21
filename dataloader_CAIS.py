import json


def read_annotation_file(file_path):
    sentences = []
    tags = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence = []
        current_tags = []
        for line in file:
            line = line.strip()
            if line:
                try:
                    word, tag = line.split()
                except:
                    word = '-'
                    tag = line.split()[0]
                current_sentence.append(word)
                current_tags.append(tag)
            else:
                if current_sentence:
                    sentences.append(''.join(current_sentence))
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
        if current_sentence:
            sentences.append(''.join(current_sentence))
            tags.append(current_tags)
    return {'asr_1best': sentences, 'tags': tags}


def read_intention_file(file_path):
    intentions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            intentions.append(line)
    return intentions


def generate_semantic(manual_transcripts, tags, intentions):
    semantic_results = []
    for i in range(len(manual_transcripts)):
        # utt_id = i + 1
        manual_transcript = manual_transcripts[i]
        current_semantic = []
        for j in range(len(tags[i])):
            tag = tags[i][j]
            if tag.startswith('B-'):
                intent = intentions[i]
                act = tag.split('-')[1]
                value = manual_transcript[j]
                k = j + 1
                while k < len(tags[i]) and tags[i][k] != 'O':
                    value += manual_transcript[k]
                    k += 1
                current_semantic.append([intent, act, value])
        semantic_results.append([{
            "utt_id": 1,
            "manual_transcript": manual_transcript,
            "asr_1best": manual_transcript,
            "semantic": current_semantic
        }])
    return semantic_results


def generate_ontology(semantic_results):
    # Return a dict of all acts and possible slots
    ontology = {"acts": set(), "slots": set()}
    for each in semantic_results:
        for each_utt in each:
            for each_semantic in each_utt['semantic']:
                act = each_semantic[0]
                slot = each_semantic[1]
                ontology['acts'].add(act)
                ontology['slots'].add(slot)
    ontology['acts'] = list(ontology['acts'])
    ontology['slots'] = list(ontology['slots'])
    return ontology


file_path = 'data/CAIS_test.txt'
intention_path = 'data/CAIS_test_intent.txt'
output_path = 'data/CAIS_test.json'
result = read_annotation_file(file_path)
intention = read_intention_file(intention_path)

semantic_results = generate_semantic(result['asr_1best'], result['tags'], intention)

ontology = generate_ontology(semantic_results)

with open(output_path, 'w') as file:
    json.dump(semantic_results, file, ensure_ascii=False, indent=4)

with open('data/CAIS_ontology.json', 'w') as file:
    json.dump(ontology, file, ensure_ascii=False, indent=4)
