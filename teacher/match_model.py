import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, model, config):
        super(BaseModel, self).__init__()
        self.bert = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        hidden, emb_cls = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.cls(emb_cls)
        return logits


class DgcnnModel(nn.Module):
    def __init__(self, model, config):
        super(DgcnnModel, self).__init__()
        self.bert = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls = nn.Linear(2 * config.hidden_size, 2)
        self.dcgnns = nn.ModuleList([DGCNNLayer(2 * config.hidden_size, 2 * config.hidden_size,
                                                k_size=item[0], dilation_rate=item[1], dropout=config.dropout_rate)
                                     for item in config.cnn_conf_list
                                     ])
        self.dropout = nn.Dropout(0.3)
        self.attention1d = AttentionPooling1D(config)
        self.concat_vec = ConcatVec()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        hidden, emb_cls = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = self.dropout(hidden)  # [batch_size, seq_length, hidden_size]
        # mask = (input_ids != 0)  # [batch_size,seq_length]
        mask = (input_ids != 0)  # [batch_size,seq_length]
        output = self.attention1d(hidden, mask)  # [batch_size, hidden_size]
        concat_output = self.concat_vec([hidden, output])  # [batch_size,seq_length, 2*hidden_size]

        out = concat_output  ## [32,256,768]
        for cnn in self.dcgnns:
            out = cnn(out, mask=mask)
        logits = self.cls(torch.max(out, dim=1)[0])
        # logits = self.sigmoid(logits)
        return logits


class DGCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, dilation_rate=1, dropout=0.1, skip_connection=True):
        super(DGCNNLayer, self).__init__()
        self.k_size = k_size
        self.dilation_rate = dilation_rate
        self.skip_connection = skip_connection
        self.hid_dim = out_channels
        self.dropout = dropout
        self.pad_size = int(self.dilation_rate * (self.k_size - 1) / 2)
        self.dropout_layer = nn.Dropout(self.dropout)
        # self.liner_layer = nn.Linear(int(out_channels / 2), out_channels)
        self.glu_layer = nn.GLU()
        self.conv_layer = nn.Conv1d(in_channels, out_channels * 2, kernel_size=k_size, dilation=dilation_rate,
                                    padding=(self.pad_size,))
        self.conv_1 = nn.Conv1d(in_channels, out_channels, 1, 1)
        self.layer_normal = nn.LayerNorm(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        '''

        :param x: shape: [batch_size, seq_length, channels(embeddings)]
        :return:
        '''
        x_r = x
        if mask is not None:
            mask = mask.unsqueeze(2)
            mask = mask.repeat(1, 1, self.hid_dim).float()
        x_r = x_r.permute(0, 2, 1)  # [batch_size, 2*hidden_size, seq_length]
        x = self.conv_layer(x_r)  # [batch_size, 2*hidden_size, seq_length]
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, 2*hidden_size]
        x, g = x[:, :, :self.hid_dim], x[:, :, self.hid_dim:]
        # x = self.glu_layer(x)  # [batch_size, seq_length, hidden_size]
        g = self.sigmoid(g)
        if self.dropout:
            g = self.dropout_layer(g)  #
        mask = mask if mask is not None else torch.ones_like(x)
        # mask = mask.unsqueeze(2).repeat(1, 1, self.hid_dim).float()
        # 残差连接
        if self.skip_connection:
            x_r = self.conv_1(x_r)
            return (x_r * (1-g) + x*g) *mask
        return x * g * mask
        # x = x * mask
        # return self.layer_normal(x + x_r)
        # return


class AttentionPooling1D(nn.Module):
    def __init__(self, config):
        super(AttentionPooling1D, self).__init__()
        self.attention_cls1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attention_cls2 = nn.Linear(config.hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(2)
        x0 = x  # [batch_size, seq_length, hidden_size]
        x = self.attention_cls1(x0)  # [batch_size, seq_length, hidden_size]
        x = self.tanh(x)  # [batch_size, seq_length, hidden_size]
        x = self.attention_cls2(x)  # [batch_size, seq_length, 1]
        x = x + torch.logical_not(mask) * -1e12
        x = self.softmax(x)  # [batch_size, seq_length, 1]  x0*x # [batch_size, seq_length, hidden_size]
        x = torch.sum(x0 * x, dim=1)  # [batch_size, hidden_size]
        return x


class ConcatVec(nn.Module):
    def __init__(self):
        super(ConcatVec, self).__init__()

    def forward(self, input):
        seq, vec = input
        vec = vec.unsqueeze(1)
        vec = vec.repeat(1, seq.shape[1], 1)  # # [batch_size,seq_length,hidden_size]
        concat_vec = torch.cat([seq, vec], dim=-1)  # [batch_size,seq_length,2*hidden_size]
        return concat_vec
        pass


if __name__ == '__main__':
    pass
