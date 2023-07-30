# coding: utf-8
from __future__ import print_function
from __future__ import division

from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from libcity.model.abstract_model import AbstractModel
from torchvision import models
from os.path import join, abspath, dirname
import math

class TPG(AbstractModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config['device']
        # depend on dataset
        self.num_neg = config['executor_config']['train']['num_negative_samples']
        self.temperature = config['executor_config']['train']['temperature']
        self.loss = config['executor_config']['loss']
        self.use_geo_encoder = config['model_config']['use_geo_encoder']
        self.pure_time_prompt = config['model_config']['pure_time_prompt']
        # from dataset
        # from train_dataset!!
        nuser = data_feature['nuser']
        nloc = data_feature['nloc']
        ntime = data_feature['ntime']
        nquadkey = data_feature['nquadkey']

        # from config
        user_dim = int(config['model_config']['user_embedding_dim'])
        loc_dim = int(config['model_config']['location_embedding_dim'])
        time_dim = int(config['model_config']['time_embedding_dim'])
        reg_dim = int(config['model_config']['region_embedding_dim'])
        # nhid = int(config['model_config']['hidden_dim_encoder'])
        nhead_enc = int(config['model_config']['num_heads_encoder'])
        # nhead_dec = int(config['model_config']['num_heads_decoder'])
        nlayers = int(config['model_config']['num_layers_encoder'])
        dropout = float(config['model_config']['dropout'])
        extra_config = config['model_config']['extra_config']
        self.use_time_query = config['model_config']['use_time_query']
        self.use_time_loss = config['model_config']['use_time_loss']
        self.loss_embedding_fusion = config['model_config']['loss_embedding_fusion']
        # print(f"nloc: {nloc} \t loc_dim: {loc_dim}")
        if extra_config.get("embedding_fusion", "multiply") == "multiply":
            ninp = user_dim
        else:
            if extra_config.get("user_embedding", False):
                # ninp = user_dim + loc_dim + time_dim + reg_dim
                ninp = loc_dim + time_dim + reg_dim
            else:
                ninp = loc_dim + reg_dim
        # essential
        self.matching_strategy = config['model_config']['matching_strategy']
        if self.matching_strategy == "mix":
            time_trg_dim = time_dim
        else:
            if self.pure_time_prompt:
                if extra_config.get("user_embedding", False):
                    # time_trg_dim = user_dim + loc_dim + time_dim + reg_dim
                    time_trg_dim = loc_dim + time_dim + reg_dim
                else:
                    time_trg_dim = loc_dim + reg_dim
            else:
                time_trg_dim = ninp 
                 
        self.clip = config['model_config']['clip']
        
        self.emb_loc = Embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.emb_time_trg = Embedding(ntime+1, time_trg_dim, zeros_pad=True, scale=True)
        self.emb_reg = Embedding(nquadkey, reg_dim, zeros_pad=True, scale=True)
        # optional
        self.emb_user = Embedding(nuser, user_dim, zeros_pad=True, scale=True)
        self.emb_time = Embedding(ntime, time_dim, zeros_pad=True, scale=True)
        if not ((user_dim == loc_dim) and (user_dim == time_dim) and (user_dim == reg_dim)):
            raise Exception("user, location, time and region should have the same embedding size!")
        
        pos_encoding = extra_config.get("position_encoding", "transformer")
        if pos_encoding == "embedding":
            self.pos_encoder = PositionalEmbedding(ninp, dropout)
        elif pos_encoding == "transformer":
            self.pos_encoder = PositionalEncoding(ninp, dropout)
        elif pos_encoding == "transformer_learnable":
            self.pos_encoder = PositionalEncodingLearnable(ninp, dropout)
        self.enc_layer = TransformerEncoderLayer(ninp, nhead_enc, ninp, dropout)
        self.encoder = TransformerEncoder(self.enc_layer, nlayers)

        self.region_pos_encoder = PositionalEmbedding(reg_dim, dropout, max_len=20)
        self.region_enc_layer = TransformerEncoderLayer(reg_dim, 1, reg_dim, dropout=dropout)
        self.region_encoder = TransformerEncoder(self.region_enc_layer, 2)

        if not extra_config.get("use_location_only", False):
            if extra_config.get("embedding_fusion", "multiply") == "concat":
                if extra_config.get("user_embedding", False):
                    self.lin = nn.Linear(user_dim + loc_dim + reg_dim + time_dim, ninp)
                else:
                    self.lin = nn.Linear(loc_dim + reg_dim, ninp)

        self.time_lin = nn.Linear(time_trg_dim, ninp)
        
        ident_mat = torch.eye(ninp)
        self.register_buffer('ident_mat', ident_mat)
        ident_mat2 = torch.eye(time_trg_dim)
        self.register_buffer('ident_mat2', ident_mat2)
        
        self.layer_norm = nn.LayerNorm(ninp)

        self.extra_config = extra_config
        self.dropout = dropout
            
    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """
        user, loc, time, region, trg, trg_reg, trg_nov, sample_probs, ds, time_query, real_time_query, time_sample_probs = batch

        assert len(region) == len(trg_reg)
        length = len(region)
        
        user = user.to(self.device)
        loc = loc.to(self.device)
        time = time.to(self.device)
        for i in range(length):
            region[i] = region[i].to(self.device)
            trg_reg[i] = trg_reg[i].to(self.device)
        trg = trg.to(self.device)
        time_query = time_query.to(self.device)
        real_time_query = real_time_query.to(self.device)
        sample_probs = sample_probs.to(self.device)
        src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(self.device) for e in ds],
                                batch_first=True, padding_value=True)
        att_mask = TPG._generate_square_mask_(max(ds), self.device)

        if self.training:
            dim = (self.num_neg + 1)*(self.num_neg + 1) if self.clip else self.num_neg + 1
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, att_mask.repeat(dim, 1), ds=ds, time_query=time_query, real_time_query=real_time_query)
        else:
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, None, ds=ds, time_query=time_query, real_time_query=real_time_query)
        return output

    @staticmethod
    def _generate_square_mask_(sz, device):
        mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def loss_function(self, output, probs, shape, ds):
        # shape: [(1+K)*L, N]
        output = output.view(-1, shape[0], shape[1]).permute(2, 1, 0)
        # shape: [N, L, 1+K]
        dim = (self.num_neg+1)*(self.num_neg+1)-1 if self.clip and self.training else self.num_neg
        pos_score, neg_score = output.split([1, dim], -1)
        if self.loss == "BinaryCELoss":
            loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) / neg_score.size(2), dim=-1)
        elif self.loss == "BPRLoss":
            loss = -F.logsigmoid(pos_score.squeeze() - neg_score.squeeze())
        elif self.loss == "WeightedBinaryCELoss":
            weight = F.softmax(neg_score / self.temperature, -1)
            loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) * weight, dim=-1)
        elif self.loss == "WeightedProbBinaryCELoss":
            weight = F.softmax(neg_score / self.temperature - torch.log(probs), -1)
            loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) * weight, dim=-1)
        keep = pad_sequence([torch.ones(e, dtype=torch.float32).to(self.device) for e in ds], batch_first=True)
        loss = torch.sum(loss * keep) / torch.sum(torch.tensor(ds).to(self.device))
        return loss
    
    def calculate_loss(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """
        # only support "WeightedProbBinaryCELoss"
        user, loc, time, region, trg, trg_reg, trg_nov, sample_probs, ds, time_query, real_time_query, time_sample_probs = batch

        assert len(region) == len(trg_reg)
        length = len(region)
        
        user = user.to(self.device)
        loc = loc.to(self.device)
        time = time.to(self.device)
        for i in range(length):
            region[i] = region[i].to(self.device)
            trg_reg[i] = trg_reg[i].to(self.device)
        trg = trg.to(self.device)
        time_query = time_query.to(self.device)
        real_time_query = real_time_query.to(self.device)
        sample_probs = sample_probs.to(self.device)
        if time_sample_probs is not None:
            time_sample_probs = time_sample_probs.to(self.device)
        src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(self.device) for e in ds],
                                batch_first=True, padding_value=True)
        att_mask = self._generate_square_mask_(max(ds), self.device)

        if self.training:
            dim = (self.num_neg + 1)*(self.num_neg + 1) if self.clip and self.training else self.num_neg + 1
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, att_mask.repeat(dim, 1), ds=ds, time_query=time_query, real_time_query=real_time_query)
        else:
            output = self.forward(user, loc, region, time, att_mask, src_mask,
                                  trg, trg_reg, None, ds=ds, time_query=time_query, real_time_query=real_time_query)

        if not self.use_time_loss or self.loss_embedding_fusion == "multiply":
            output = output[0]
            loss = self.loss_function(output, sample_probs, loc.shape, ds)
        elif self.loss_embedding_fusion == "mean":
            output1, output2 = output
            loss1 = self.loss_function(output1, sample_probs, loc.shape, ds)
            loss2 = self.loss_function(output2, time_sample_probs, loc.shape, ds)
            loss = (loss1 + loss2) / 2
            
        return loss
    
    def forward(self, src_user, src_loc, src_reg, src_time,
                src_square_mask, src_binary_mask, trg_loc, trg_reg, mem_mask, ds=None, time_query=None, real_time_query=None):
        length = len(src_reg)
        loc_emb_src = self.emb_loc(src_loc)
        if self.extra_config.get("user_location_only", False):
            src = loc_emb_src
        else:
            user_emb_src = self.emb_user(src_user)
            # (L, N, LEN_QUADKEY, REG_DIM) eg: 109, 64, 12, 50
            reg_emb = torch.mean(torch.cat([self.emb_reg(src_reg[i]).unsqueeze(0) for i in range(length)], dim=0), dim=0)
            reg_emb = reg_emb.view(reg_emb.size(0) * reg_emb.size(1),
                                   reg_emb.size(2), reg_emb.size(3)).permute(1, 0, 2)
            # (LEN_QUADKEY, L * N, REG_DIM)
            
            reg_emb = self.region_pos_encoder(reg_emb)
            reg_emb = self.region_encoder(reg_emb)
            # avg pooling
            reg_emb = torch.mean(reg_emb, dim=0)

            # reg_emb, _ = self.region_gru_encoder(reg_emb, self.h_0.expand(4, reg_emb.size(1), -1).contiguous())
            # reg_emb = reg_emb[-1, :, :]

            # (L, N, REG_DIM)
            reg_emb = reg_emb.view(loc_emb_src.size(0), loc_emb_src.size(1), reg_emb.size(1))

            time_emb = self.emb_time(src_time)
            if self.extra_config.get("embedding_fusion", "multiply") == "multiply":
                if self.extra_config.get("user_embedding", False):
                    src = loc_emb_src * reg_emb * time_emb * user_emb_src
                else:
                    src = loc_emb_src * reg_emb * time_emb
            else:
                if self.extra_config.get("user_embedding", False):
                    # src = torch.cat([user_emb_src, loc_emb_src, reg_emb, time_emb], dim=-1)
                    src = torch.cat([loc_emb_src, reg_emb, time_emb], dim=-1)
                else:
                    src = torch.cat([loc_emb_src, reg_emb], dim=-1)
 
        if self.extra_config.get("size_sqrt_regularize", True):
            src = src * math.sqrt(src.size(-1))

        src = self.pos_encoder(src)
        # shape: [L, N, ninp]
        src = self.encoder(src, mask=src_square_mask)

        # shape: [(1+K)*L, N, loc_dim]
        loc_emb_trg = self.emb_loc(trg_loc)

        reg_emb_trg = torch.mean(torch.cat([self.emb_reg(trg_reg[i]).unsqueeze(0) for i in range(length)], dim=0), dim=0)  # [(1+K)*L, N, LEN_QUADKEY, REG_DIM]
        # (LEN_QUADKEY, (1+K)*L * N, REG_DIM)
        reg_emb_trg = reg_emb_trg.view(reg_emb_trg.size(0) * reg_emb_trg.size(1),
                                       reg_emb_trg.size(2), reg_emb_trg.size(3)).permute(1, 0, 2)
        reg_emb_trg = self.region_pos_encoder(reg_emb_trg)
        reg_emb_trg = self.region_encoder(reg_emb_trg)
        reg_emb_trg = torch.mean(reg_emb_trg, dim=0)
        # [(1+K)*L, N, REG_DIM]
        reg_emb_trg = reg_emb_trg.view(loc_emb_trg.size(0),
                                       loc_emb_trg.size(1), reg_emb_trg.size(1))

        # shape: [(1+K)*L, N, ninp]
        time_emb_trg = self.emb_time_trg(time_query)
        time_emb_trg = time_emb_trg.repeat(loc_emb_trg.size(0) // time_emb_trg.size(0), 1, 1)
            
        if self.extra_config.get("embedding_fusion", "multiply") == "multiply":
            loc_emb_trg = loc_emb_trg * reg_emb_trg
        else:
            if self.extra_config.get("user_embedding", False):
                if not self.training:
                    user_emb_src = user_emb_src[0].unsqueeze(0)
                if not self.clip or not self.training:
                    # loc_emb_trg = torch.cat([user_emb_src.repeat(loc_emb_trg.size(0) // user_emb_src.size(0), 1, 1), loc_emb_trg, reg_emb_trg, time_emb_trg], dim=-1)
                    loc_emb_trg = torch.cat([loc_emb_trg, reg_emb_trg, time_emb_trg], dim=-1)
                else:
                    seq_length = loc_emb_trg.size(0) // (self.num_neg+1)
                    for i in range(self.num_neg+1):
                        loc_query = torch.cat([loc_emb_trg[i*seq_length:(i+1)*seq_length], reg_emb_trg[i*seq_length:(i+1)*seq_length]], dim=-1)
                        for j in range(self.num_neg+1):
                            tmp = torch.cat([loc_query, time_emb_trg[j*seq_length:(j+1)*seq_length]], dim=-1)
                            if i==0 and j==0:
                                clip_emb = tmp
                            else:
                                clip_emb = torch.cat([clip_emb, tmp], dim=0)
                    # loc_emb_trg = torch.cat([user_emb_src.repeat(clip_emb.size(0) // user_emb_src.size(0), 1, 1), clip_emb], dim=-1)
                    loc_emb_trg = clip_emb
            else:
                loc_emb_trg = torch.cat([loc_emb_trg, reg_emb_trg], dim=-1)
        
        src2 = src.clone()
        if self.extra_config.get("use_attention_as_decoder", False):
            # multi-head attention
            output, _ = F.multi_head_attention_forward(
                query=time_emb_trg if self.pure_time_prompt else loc_emb_trg,
                key=src,
                value=src,
                embed_dim_to_check=src.size(2),
                num_heads=1,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=None,
                dropout_p=0.0,
                out_proj_weight=self.ident_mat,
                out_proj_bias=None,
                training=self.training,
                key_padding_mask=src_binary_mask,
                need_weights=False,
                attn_mask=mem_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.ident_mat,
                k_proj_weight=self.ident_mat,
                v_proj_weight=self.ident_mat
            )

            if self.training:
                src = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                src = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                src = src.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

            output += src
            output = self.layer_norm(output)
            
            if self.matching_strategy == "poi_time":
                # multi-head attention
                output2, _ = F.multi_head_attention_forward(
                    query=time_emb_trg,
                    key=src2,
                    value=src2,
                    embed_dim_to_check=src2.size(2),
                    num_heads=1,
                    in_proj_weight=None,
                    in_proj_bias=None,
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=None,
                    dropout_p=0.0,
                    out_proj_weight=self.ident_mat2,
                    out_proj_bias=None,
                    training=self.training,
                    key_padding_mask=src_binary_mask,
                    need_weights=False,
                    attn_mask=mem_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.ident_mat2,
                    k_proj_weight=self.ident_mat2,
                    v_proj_weight=self.ident_mat2
                )
                if self.training:
                    src2 = src2.repeat(time_emb_trg.size(0) // src2.size(0), 1, 1)
                else:
                    src2 = src2[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                    src2 = src2.unsqueeze(0).repeat(time_emb_trg.size(0), 1, 1)

                output2 += src2
                output2 = self.layer_norm(output2)
                output = self.layer_norm((output + output2)/2)
        else:
            # No attention
            if self.training:
                output = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                output = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                output = output.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

        # shape: [(1+K)*L, N]
        temp = output.clone()
        output = torch.sum(output * loc_emb_trg, dim=-1)
        if not self.use_time_loss:
            return [output, None]
        else:
            if self.loss_embedding_fusion == "mean":
                time_output = torch.sum(temp * time_emb_trg, dim=-1)
                return [output, time_output]
            elif self.loss_embedding_fusion == "multiply":
                mix_output = torch.sum(temp * loc_emb_trg * time_emb_trg, dim=-1)
                return [mix_output, None]
            else:
                raise ValueError("The type of loss embedding {} is not correct!".format(self.loss_embedding_fusion))
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(
            inputs, self.lookup_table,
            self.padding_idx, None, 2, False, False)  # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb_table = Embedding(max_len, d_model, zeros_pad=False, scale=False)
        pos_vector = torch.arange(max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_vector', pos_vector)

    def forward(self, x):
        pos_emb = self.pos_emb_table(self.pos_vector[:x.size(0)].unsqueeze(1).repeat(1, x.size(1)))
        x += pos_emb
        return self.dropout(x)

class PositionalEncodingLearnable(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLearnable, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze*0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        