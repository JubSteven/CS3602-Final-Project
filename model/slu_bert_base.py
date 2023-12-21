import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM
import jieba
from text2vec import SentenceModel
import logging
import numpy as np
from torch.nn import Transformer
# from accelerate import Accelerator

# accelerator = Accelerator()

nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        # print("num_tags = ", num_tags) , 74

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += mask.float().unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob,)


class RNNTaggingDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id, model_type="GRU", num_layers=1):
        super(RNNTaggingDecoder, self).__init__()
        assert model_type in ["LSTM", "GRU", "RNN"], 'model_type should be one of "LSTM", "GRU", "RNN"'
        self.num_tags = num_tags
        self.feat_dim = 100
        self.output_layer = getattr(nn, model_type)(input_size,
                                                    self.feat_dim,
                                                    num_layers=num_layers,
                                                    bidirectional=True)
        self.linear_layer = nn.Linear(self.feat_dim * 2, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        # print("num_tags = ", num_tags) , 74

    def forward(self, hiddens, mask, labels=None):
        logits, _ = self.output_layer(hiddens)
        logits = self.linear_layer(logits)
        logits += mask.float().unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
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

        if bertConfig.LA_decoder:
            self.hidden_decode = getattr(nn, bertConfig.LA_decoder)(bertConfig.hidden_size,
                                                                    bertConfig.hidden_size,
                                                                    bidirectional=False,
                                                                    batch_first=True)

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
        # model, char_word_pair = Accelerator.prepare(self.text_embed, char_word_pair)
        model = self.text_embed
        # print(char_word_pair)
        words_to_encode = [word for pair_list in char_word_pair for pair in pair_list for word in pair if word]
        # 批量编码
        encoded_words = model.encode(words_to_encode)

        # 处理空字符串的情况
        default_vector = model.encode("")  # 假设模型有一个方法来获取嵌入维度
        # print(default_vector)

        # 将编码结果分配回原始数据结构
        idx = 0
        for pair_list in char_word_pair:
            for pair in pair_list:
                for i in range(len(pair)):
                    if pair[i]:
                        pair[i] = encoded_words[idx]
                        idx += 1
                    else:
                        pair[i] = default_vector

        # 转换为张量
        embedded_char_word_pair = torch.Tensor(np.array(char_word_pair))

        # print(embedded_char_word_pair.shape)
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
        if self.hidden_decode:
            layer_output, _ = self.hidden_decode(layer_output)

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

    def __init__(self, config):
        super(SLUFusedBertTagging, self).__init__()
        self.config = config
        # embed_size = 768
        # hidden_size = 512
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.tag_embed = nn.Embedding(config.num_tags, config.embed_size)
        self.transformer = Transformer(d_model=config.embed_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=config.dropout,
                                       batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def positional_encoding(self, X, num_features, dropout_p=0.1, max_len=512):
        r'''
            给输入加入位置编码
        参数：
            - num_features: 输入进来的维度
            - dropout_p: dropout的概率，当其为非零时执行dropout
            - max_len: 句子的最大长度，默认512
        形状：
            - 输入： [batch_size, seq_length, num_features]
            - 输出： [batch_size, seq_length, num_features]

        例子：
            >>> X = torch.randn((2,4,10))
            >>> X = positional_encoding(X, 10)
            >>> print(X.shape)
            >>> torch.Size([2, 4, 10])
        '''

        dropout = nn.Dropout(dropout_p)
        P = torch.zeros((1, max_len, num_features))
        X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
        P[:, :, 0::2] = torch.sin(X_)
        P[:, :, 1::2] = torch.cos(X_)
        X = X + P[:, :X.shape[1], :].to(X.device)
        return dropout(X)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = (1 - batch.tag_mask).bool()
        input_ids = batch.input_ids
        lengths = batch.lengths

        src_embed = self.word_embed(input_ids)
        src_embed = self.positional_encoding(X=src_embed, num_features=self.config.embed_size)
        tgt_embed = self.tag_embed(tag_ids)
        tgt_embed = self.positional_encoding(X=tgt_embed, num_features=self.config.embed_size)

        mask = self.transformer.generate_square_subsequent_mask(
            max(lengths)).to("cuda" if torch.cuda.is_available() else "cpu")
        # packed_inputs = rnn_utils.pack_padded_sequence(src_embed, lengths, batch_first=True)
        # packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        # rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        outs = self.transformer(src=src_embed,
                                tgt=tgt_embed,
                                tgt_mask=mask,
                                src_key_padding_mask=tag_mask,
                                tgt_key_padding_mask=tag_mask)
        # print(outs.shape)
        hiddens = self.dropout_layer(outs)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
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
        return predictions, labels, loss.cpu().item()
