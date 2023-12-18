import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator


class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, cfg):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):  # utt is a dict here
                ex = cls(utt, f'{di}-{ui}', cfg)
                examples.append(ex)
        return examples

    def _get_val_indx(self, utt, target):
        idx_list = []
        targ_idx = 0
        for i in range(len(utt)):
            if targ_idx < len(target) and utt[i] == target[targ_idx]:
                targ_idx += 1
                idx_list.append(i)

        return idx_list

    def __init__(self, ex: dict, did, cfg):
        super(Example, self).__init__()
        self.ex = ex  # a dict containing the data associated with one sentence
        self.did = did  # data id

        self.utt = ex['asr_1best']  # the sentence
        self.gt = ex["manual_transcript"]  # record the ground truth
        self.flexible_tag = cfg.flexible_tag

        if cfg.use_gt:
            self.utt = self.gt

        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:  # to prevent the case where the label's last element is empty
                self.slot[act_slot] = label[2]
        # 'O' serves as a separator, 'B' symbols the start and 'I' symbols mid-word. This is used for POS-Tagging later.
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]

            if self.flexible_tag:
                bidx = self._get_val_indx(self.utt, value)
                if bidx:
                    self.tags = [f'I-{slot}' if i in bidx else 'O' for i in range(len(self.utt))]
                    self.tags[bidx[0]] = f'B-{slot}'
            else:
                bidx = self.utt.find(value)  # from the beginning of the sentence, find the index of the value
                if bidx != -1:
                    self.tags[bidx:bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'

        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()
                         ]  # might have multiple slot-value pairs
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
