import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from utils.args import args as arguments


class BIO(Enum):
    B = 0
    I = 1
    O = 2


@dataclass
class Label:
    bio: BIO
    act: str
    slot: str


@dataclass
class Sentence:
    vector_with_noise: torch.Tensor
    vector_without_noise: torch.Tensor
    tokens_with_noise: List[str]
    tokens_without_noise: List[str]


class LabelConverter:
    def __init__(self, ontology_path):
        with open(ontology_path, encoding='utf-8') as f:
            data = json.load(f)
        self.act_to_index = {v: i for i, v in enumerate(data['acts'])}
        self.index_to_act = {i: v for i, v in enumerate(data['acts'])}
        self.slot_to_index = {v: i for i, v in enumerate(data['slots'])}
        self.index_to_slot = {i: v for i, v in enumerate(data['slots'])}
        self.n_acts = len(self.act_to_index)
        self.n_slots = len(self.slot_to_index)

    def label_to_index(self, label: Label) -> int:
        if label.bio == BIO.O:
            return 2 * self.n_acts * self.n_slots
        i = 0 if label.bio == BIO.B else 1
        i = i * self.n_acts + self.act_to_index[label.act]
        i = i * self.n_slots + self.slot_to_index[label.slot]
        return i

    def index_to_label(self, i: int) -> Label:
        if i == 2 * self.n_acts * self.n_slots:
            return Label(BIO.O, '', '')
        slot_idx = i % self.n_slots
        i //= self.n_slots
        act_idx = i % self.n_acts
        i //= self.n_acts
        bio = BIO.B if i == 0 else BIO.I
        return Label(bio, self.index_to_act[act_idx], self.index_to_slot[slot_idx])

    @property
    def num_indexes(self) -> int:
        return 2 * self.n_acts * self.n_slots + 1


class MyDataset(Dataset):
    pattern = re.compile(r'\(.*\)')

    def __init__(self, data_path, label_converter: LabelConverter, model_name: str, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        md5 = hashlib.md5((model_name + data_path + cache_dir).encode('utf-8')).hexdigest()
        cache_path = Path(cache_dir) / md5
        if cache_path.is_file():
            self._data = torch.load(cache_path, map_location=arguments.device)
            return
        self.label_converter = label_converter
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        # output_len = list(model.parameters())[-1].shape[0]
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
        self._data = []
        for i in data:
            n_utt = len(i)
            features = [None] * n_utt
            labels = [None] * n_utt
            for j in i:
                text_with_noise = j['asr_1best']
                text_without_noise = j['manual_transcript']
                # remove special tokens in the dataset, i.e., (unknown)
                text_with_noise = re.sub(self.pattern, '', text_with_noise)
                text_without_noise = re.sub(self.pattern, '', text_without_noise)
                k = j['utt_id'] - 1
                model_input = tokenizer(text_with_noise, return_tensors='pt')
                output = model(**model_input)
                vector_with_noise = output[0][0].to(arguments.device).detach()
                model_input = tokenizer(text_without_noise, return_tensors='pt')
                output = model(**model_input)
                vector_without_noise = output[0][0].to(arguments.device).detach()
                tokens_with_noise = tokenizer.tokenize(text_with_noise)
                tokens_without_noise = tokenizer.tokenize(text_without_noise)
                features[k] = Sentence(vector_with_noise, vector_without_noise, tokens_with_noise, tokens_without_noise)
                bio_labels = self._get_bio_labels(tokens_with_noise, j['semantic'])
                tensor = torch.zeros([len(bio_labels), self.label_converter.num_indexes])
                for i2, v2 in enumerate(bio_labels):
                    tensor[i2, v2] = 1
                labels[k] = tensor.to(arguments.device)
            self._data.append((features, labels))
        torch.save(self._data, cache_path)
        print('The cache is written. Please run again.')
        sys.exit(0)

    def _get_bio_labels(self, text: List[str], labels: List[Tuple[str, str, str]]) -> List[int]:
        # text 是 tokens with noise，是从asr中由bert tokenizer拿到的
        # labels 则是数据集中的semantics，它是原本的labels来源
        labels = [i for i in labels]

        # make labels appear in the same order as they appear in the text
        labels = self._rearrange_labels(''.join(text), labels)

        ret = [self.label_converter.label_to_index(Label(BIO.O, '', ''))] * (len(text) + 2)
        if len(labels) == 0:
            return ret
        j = 0
        act, slot, value = labels[j] # semantics里的三元组，value有可能不在asr的text里
        begin = True
        for i, v in enumerate(text): # 这里每个v是token，注意不是单个字 ⚠️
            if value.startswith(v):
                # 如果asr句子中，这个字以它开始，那么就会算做标记，而不是像原来的数据处理一样，在句子中先找到一整个value，对整个value做标记
                # 比如:
                """
                    manual: 这个｜大作业｜的｜数据处理｜很有｜问题｜啊
                    asr：这个｜大作业｜的｜数据处理｜很有｜问｜啊
                    如果semantics里有一个(slot-act-value: (B-act1-问)(I-act1-题)，然后它tokenize的时候正好asrbest里有“问”这个token：
                    - official：它的label里，”问“ 的slot-act那个应该 O，代表缺失
                    - This：它的label里， ”问“ 的slot-act那个应该是（B-act1-问），增加了一个不为空的label
                    由于bert的强大，它肯定会给“问”打上label，从而在official的case里，降低正确率；它这个方法利用了这一点，增加了正确率。
                """
                bio = BIO.B if begin else BIO.I
                ret[i + 1] = self.label_converter.label_to_index(Label(bio, act, slot))
                begin = False
                value = value[len(v):]
            if len(value) == 0:
                j += 1
                if j >= len(labels):
                    break
                act, slot, value = labels[j]
                begin = True
        if j < len(labels):
            raise RuntimeError('The text and tags are inconsistent.')
        return ret

    @staticmethod
    def _rearrange_labels(text: str, labels: List[Tuple[str, str, str]]):
        labels = sorted(labels, key=lambda x: len(x[2]), reverse=True)
        for i, v in enumerate(labels):
            j = text.find(v[2])
            labels[i].append(j)
            text = text.replace(v[2], '#' * len(v[2]), 1)
        labels = [i for i in labels if i[3] != -1]
        labels = sorted(labels, key=lambda x: x[3])
        labels = [i[:3] for i in labels]
        return labels

    def __getitem__(self, index: int) -> Tuple[List[Sentence], List[torch.Tensor]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)


class MyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self._my_collate_func
        super().__init__(*args, **kwargs)

    @staticmethod
    def _my_collate_func(batch):
        return tuple(zip(*batch))

    def __iter__(self) -> Iterator[Tuple[List[List[Sentence]], List[List[torch.Tensor]]]]:
        '''Just for type hints.'''
        return super().__iter__()
