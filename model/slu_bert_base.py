import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM


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
    
    def __init__(self, config):
        super(LexionAdapter, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        
        self.word_transform = nn.Linear(config.word_embed_dim, config.hidden_size)
        self.word_word_weight = nn.Linear(config.hidden_size, config.hidden_size)
        attn_W = torch.zeros(config.hidden_size, config.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)
        self.fuse_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    
    def forward(self, input_word_embeddings, layer_output, input_word_mask=None):
        """
            Inputs:
            input_word_embeddings: 
            layer_output: 
        """
        
        # transform
        word_outputs = self.word_transform(input_word_embeddings)  # [N, L, W, D]
        word_outputs = self.tanh(word_outputs)
        word_outputs = self.word_word_weight(word_outputs)
        word_outputs = self.dropout(word_outputs)

        # attention_output = attention_output.unsqueeze(2) # [N, L, D] -> [N, L, 1, D]
        alpha = torch.matmul(layer_output.unsqueeze(2), self.attn_W)  # [N, L, 1, D]
        alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))  # [N, L, 1, W]
        alpha = alpha.squeeze()  # [N, L, W]
        
        # ! I don't know what it is used for. Input_word_mask should be OK for None.
        alpha = alpha + (1 - input_word_mask.float()) * (-10000.0)
        
        alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, L, W]
        alpha = alpha.unsqueeze(-1)  # [N, L, W, 1]
        
        weighted_word_embedding = torch.sum(word_outputs * alpha, dim=2)  # [N, L, D]
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

        self.output_layer = TaggingFNNDecoder(self.hidden_size, self.num_tags, cfg.tag_pad_idx)

    def set_model(self):
        # assert self.model_type in ["bert-base-chinese", "MiniRBT-h256-pt", "MacBERT-base","MacBERT-large"]
        if self.model_type == "bert-base-chinese":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_type)
            self.model = BertModel.from_pretrained(self.model_type, output_hidden_states=True).to(self.device)
        elif self.model_type == "MacBERT-base":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base", output_hidden_states=True).to(
                self.device)
        elif self.model_type == "MacBERT-large":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-large",
                                                              output_hidden_states=True).to(self.device)
        elif self.model_type == "roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base", output_hidden_states=True).to(
                self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_type, output_hidden_states=True).to(
                self.device)
        
        self.config = BertConfig.from_pretrained(self.model_type)

    def forward(self, batch):
        """
            Here batch is a list of original sentences
        """
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask

        self.length = [len(utt) for utt in batch.utt]
        encoded_inputs = self.tokenizer(batch.utt, padding="max_length", truncation=True,
                                        max_length=max(self.length), return_tensors='pt').to(self.device)

        # hiddens = self.model(**encoded_inputs).last_hidden_state
        hiddens = self.model(**encoded_inputs).hidden_states[-1]

        # Return the padded sentence (shape)
        # [B, MAX_LENGTH, F]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids) # tagoutput[0] is the prob, tagoutput[1] is the loss
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
            pred = pred[:len(batch.utt[i])] # the length of the sentence i in the batch, like [61, 2, 55, 14, 7, 63, 26, 42, 32, 32, 15, 29, 21, 15, 52, 52, 21, 6, 26, 42]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid) # ex: I-deny-出行方式
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
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-base", output_hidden_states=True).to(
                self.device)
        elif self.model_type == "MacBERT-large":
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
            self.model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-macbert-large",
                                                              output_hidden_states=True).to(self.device)
        elif self.model_type == "roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base", output_hidden_states=True).to(
                self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_type, output_hidden_states=True).to(
                self.device)
        
        self.config = BertConfig.from_pretrained(self.model_type)

    def forward(self, batch):
        """
            Here batch is a list of original sentences
        """
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask

        self.length = [len(utt) for utt in batch.utt]
        encoded_inputs = self.tokenizer(batch.utt, padding="max_length", truncation=True,
                                        max_length=max(self.length), return_tensors='pt').to(self.device)

        # hiddens = self.model(**encoded_inputs).last_hidden_state
        hiddens = self.model(**encoded_inputs).hidden_states[-1]

        # Return the padded sentence (shape)
        # [B, MAX_LENGTH, F]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids) # tagoutput[0] is the prob, tagoutput[1] is the loss
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
            pred = pred[:len(batch.utt[i])] # the length of the sentence i in the batch, like [61, 2, 55, 14, 7, 63, 26, 42, 32, 32, 15, 29, 21, 15, 52, 52, 21, 6, 26, 42]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid) # ex: I-deny-出行方式
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
