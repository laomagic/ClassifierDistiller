import torch.nn as nn
import torch
from transformers import BertModel


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def initializer_builder(std):
    _std = std

    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights


class BertClassifier(nn.Module):
    def __init__(self, config, num_labels, args):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        self.output_hidden_states = args.output_hidden_states
        self.output_attentions = args.output_attentions
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask,
                            output_attentions=self.output_attentions,
                            output_hidden_states=self.output_hidden_states)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
