import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM
import jieba
from text2vec import SentenceModel
import logging
import numpy as np


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        # print("num_tags = ", num_tags) , 74

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob,)


# >>>>>>@pengxiang added the Module on 12.14
class LexionAdapter(nn.Module):
    """
        A fusion model that takes in a charactor vector and the paired word features.
        Adapted from https://github.com/liuwei1206/LEBERT/blob/main/wcbert_modeling.py
    """

    def __init__(self, bertConfig):
        super(LexionAdapter, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

        self.text_embed = SentenceModel()

        self.word_transform = nn.Linear(bertConfig.word2vec_embed_size, bertConfig.hidden_size)
        self.word_word_weight = nn.Linear(bertConfig.hidden_size, bertConfig.hidden_size)
        attn_W = torch.zeros(bertConfig.hidden_size, bertConfig.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=bertConfig.initializer_range)
        self.fuse_layernorm = nn.LayerNorm(bertConfig.hidden_size, eps=bertConfig.layer_norm_eps)

    # >>>>>>@pengxiang added on 12.15
    def _char_word_pair(self, batch_sentences, pad_length=None):
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

    def _word_embed(self, char_word_pair):
        """Takes in original char_word_pairs and transform them into vectors using text2vec sentence model.

        Args:
            char_word_pair (list [list [list]]): [[["美国"], ["美国"], ["人民"], ["人民"], [""]] ..]
        """

        for pair_list in char_word_pair:
            for pair in pair_list:
                for i in range(len(pair)):
                    pair[i] = self.text_embed.encode(pair[i])

        embedded_char_word_pair = torch.Tensor(np.array(char_word_pair))
        return embedded_char_word_pair

    def process_sentence(self, input_sentence):
        """
            Inputs:
            input_sentence: a list of length B that stores original Chinese sentences in a batch
        """
        input_char_word_pair = self._char_word_pair(input_sentence)
        word_embed = self._word_embed(input_char_word_pair)
        return word_embed

    def forward(self, input_sentence, layer_output):
        """
            Inputs:
            input_sentence: a list of length B that stores original Chinese sentences in a batch
            layer_output: The result of a hidden layer of size [B, L, D] where L is the padded length and D is the feature dim.
        """
        input_word_embeddings = self.process_sentence(input_sentence)  # [B, ?] -> [B, L, W, D]
        input_word_embeddings = input_word_embeddings.to(layer_output.device)

        # transform
        word_outputs = self.word_transform(input_word_embeddings)  # [B, L, W, D]
        word_outputs = self.tanh(word_outputs)
        word_outputs = self.word_word_weight(word_outputs)
        word_outputs = self.dropout(word_outputs)

        alpha = torch.matmul(layer_output.unsqueeze(2), self.attn_W)  # [B, L, 1, D]
        alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))  # [B, L, 1, W]
        alpha = alpha.squeeze()  # [B, L, W]

        # ! I don't know what it is used for. Input_word_mask should be OK for None.
        # alpha = alpha + (1 - input_word_mask.float()) * (-10000.0)

        alpha = torch.nn.Softmax(dim=-1)(alpha)  # [B, L, W]
        alpha = alpha.unsqueeze(-1)

        weighted_word_embedding = torch.sum(word_outputs * alpha, dim=2)  # [B, L, D]
        layer_output = layer_output + weighted_word_embedding

        layer_output = self.dropout(layer_output)
        layer_output = self.fuse_layernorm(layer_output)

        return layer_output


class SLUFusedBertTagging(nn.Module):

    def __init__(self, cfg):
        super(SLUFusedBertTagging, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.num_tags = cfg.num_tags
        self.model_type = cfg.encoder_cell
        self.set_model()

        # TODO can polish a bit here
        if self.model_type == "bert-base-chinese" or "MacBERT-base" or "roberta-base":
            self.hidden_size = 768
        else:
            self.hidden_size = 256

        # ! Do not change the name LA_layer, in sync with slu_fused_bert.py
        self.LA_layer = LexionAdapter(self.bertConfig)
        self.output_layer = TaggingFNNDecoder(self.hidden_size, self.num_tags, cfg.tag_pad_idx)

    def set_model(self):
        # assert self.model_type in ["bert-base-chinese", "MiniRBT-h256-pt", "MacBERT-base","MacBERT-large"]
        if self.model_type == "bert-base-chinese":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_type)
            self.model = BertModel.from_pretrained(self.model_type, output_hidden_states=True).to(self.device)
        elif self.model_type == "MacBERT-base":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base",
                                                              output_hidden_states=True).to(self.device)
        elif self.model_type == "MacBERT-large":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-large",
                                                              output_hidden_states=True).to(self.device)
        elif self.model_type == "roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base",
                                                   output_hidden_states=True).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_type,
                                                              output_hidden_states=True).to(self.device)

        self.bertConfig = BertConfig.from_pretrained(self.model_type)

        # Similar to the baseline, add the key attributes here.
        self.bertConfig.word2vec_embed_size = self.cfg.embed_size
        self.bertConfig.word2vec_vocab_size = self.cfg.vocab_size

    def forward(self, batch):
        """
            Here batch is a list of original sentences
        """
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_word2vec_ids = batch.input_ids  # already padded

        self.length = [len(utt) for utt in batch.utt]
        encoded_inputs = self.tokenizer(batch.utt,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=max(self.length),
                                        return_tensors='pt').to(self.device)

        hiddens = self.model(**encoded_inputs).hidden_states[-1]

        fused_hiddens = self.LA_layer(batch.utt, hiddens)

        # Return the padded sentence (shape)
        # [B, MAX_LENGTH, D]
        tag_output = self.output_layer(fused_hiddens, tag_mask,
                                       tag_ids)  # tagoutput[0] is the prob, tagoutput[1] is the loss
        # print(tag_output[0].shape) # 32(batch_size)， 26(utt length)， 74(num_tags))
        print("Finish forward")
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []

        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # batch.utt[i] composed of a list of sentences
            pred = pred[:len(
                batch.utt[i]
            )]  # the length of the sentence i in the batch, like [61, 2, 55, 14, 7, 63, 26, 42, 32, 32, 15, 29, 21, 15, 52, 52, 21, 6, 26, 42]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)  # ex: I-deny-出行方式
                pred_tags.append(tag)
                # 'O' serves as a separator, 'B' symbols the start and 'I' symbols mid-word. This is used for POS-Tagging later.
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)

        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class SLUNaiveBertTagging(nn.Module):

    def __init__(self, cfg):
        super(SLUNaiveBertTagging, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.num_tags = cfg.num_tags
        self.model_type = cfg.encoder_cell

        self.set_model()

        # TODO can polish a bit here
        if self.model_type == "bert-base-chinese" or "MacBERT-base" or "roberta-base":
            self.hidden_size = 768
        else:
            self.hidden_size = 256

        self.output_layer = TaggingFNNDecoder(self.hidden_size, self.num_tags, cfg.tag_pad_idx)

    def set_model(self):
        # assert self.model_type in ["bert-base-chinese", "MiniRBT-h256-pt", "MacBERT-base","MacBERT-large"]
        if self.model_type == "bert-base-chinese":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_type)
            self.model = BertModel.from_pretrained(self.model_type, output_hidden_states=True).to(self.device)
        elif self.model_type == "MacBERT-base":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base",
                                                              output_hidden_states=True).to(self.device)
        elif self.model_type == "MacBERT-large":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-large",
                                                              output_hidden_states=True).to(self.device)
        elif self.model_type == "roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base",
                                                   output_hidden_states=True).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_type,
                                                              output_hidden_states=True).to(self.device)

        self.config = BertConfig.from_pretrained(self.model_type)

    def forward(self, batch):
        """
            Here batch is a list of original sentences
        """
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask

        self.length = [len(utt) for utt in batch.utt]
        encoded_inputs = self.tokenizer(batch.utt,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=max(self.length),
                                        return_tensors='pt').to(self.device)

        # hiddens = self.model(**encoded_inputs).last_hidden_state
        hiddens = self.model(**encoded_inputs).hidden_states[-1]

        # Return the padded sentence (shape)
        # [B, MAX_LENGTH, F]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)  # tagoutput[0] is the prob, tagoutput[1] is the loss
        # print(tag_output[0].shape) # 32(batch_size)， 26(utt length)， 74(num_tags))
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []

        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # batch.utt[i] composed of a list of sentences
            pred = pred[:len(
                batch.utt[i]
            )]  # the length of the sentence i in the batch, like [61, 2, 55, 14, 7, 63, 26, 42, 32, 32, 15, 29, 21, 15, 52, 52, 21, 6, 26, 42]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)  # ex: I-deny-出行方式
                pred_tags.append(tag)
                # 'O' serves as a separator, 'B' symbols the start and 'I' symbols mid-word. This is used for POS-Tagging later.
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)

        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()
