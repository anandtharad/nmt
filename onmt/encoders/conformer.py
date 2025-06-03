import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward, ActivationFunction
from onmt.utils.misc import sequence_mask

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.layer_norm = nn.LayerNorm(d_model)
        #self.batch_norm = nn.BatchNorm1d(d_model, track_running_stats=False)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        #x = self.batch_norm(x) # Potentially change to LayerNorm 
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout, 
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            add_ffnbias=False,
            norm_eps=1e-6,
        )
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout, is_decoder=False
        )
        self.conv_module = ConformerConvModule(d_model, dropout=dropout)
        self.ffn2 = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            add_ffnbias=False,
            norm_eps=1e-6,
        )

        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_mha = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm_ffn1(x)
        x = residual + 0.5 * self.dropout(self.ffn1(x))

        residual = x
        x = self.norm_mha(x)
        x_attn, _ = self.self_attn(x, x, x, mask=mask)
        x = residual + self.dropout(x_attn)

        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x))

        residual = x
        x = self.norm_ffn2(x)
        x = residual + 0.5 * self.dropout(self.ffn2(x))

        return x


class ConformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super().__init__()
        self.embeddings = embeddings
        self.layers = nn.ModuleList([
            ConformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                pos_ffn_activation_fn=pos_ffn_activation_fn
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    @classmethod
    def from_opt(cls, opt, embeddings):
        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if isinstance(opt.dropout, list) else opt.dropout,
            opt.attention_dropout[0] if isinstance(opt.attention_dropout, list) else opt.attention_dropout,
            embeddings,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
        )

    def forward(self, src, src_len=None):
        x = self.embeddings(src)
        mask = sequence_mask(src_len).unsqueeze(1).unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        return x, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.layers:
            layer.dropout.p = dropout
