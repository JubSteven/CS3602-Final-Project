import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModel


class BertHiddenGroupFusion(nn.Module):
    def __init__(self):
        super(BertHiddenGroupFusion, self).__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768 * 4, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.residual = nn.Linear(768, 768)

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, 8, 26, 768, device=x.device)

        for i in range(8):
            group = x[:, i * 4:(i + 1) * 4, :, :]
            group = group.view(batch_size, 26, -1)
            fused = self.fusion_mlp(group)

            # 残差连接
            residual = self.residual(fused)
            fused = fused + residual

            # 自注意力机制
            fused = fused.permute(1, 0, 2)  # 调整维度以适应自注意力模块
            attn_output, _ = self.self_attention(fused, fused, fused)
            attn_output = attn_output.permute(1, 0, 2)  # 恢复原始维度

            output[:, i, :, :] = attn_output
        return output


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

        print(len(hiddens))

        for h in hiddens:
            print(h.shape)
            
        exit(0)

        # Return the padded sentence (shape)
        # [B, MAX_LENGTH, F]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids) # tagoutput[0] is the prob, tagoutput[1] is the loss
        # print(tag_output[0].shape) # 32(batch_size)， 26(utt length)， 74(num_tags))
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0] # first element of output is the prob
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
