import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


class GPTForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes=2):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        self.dropout_rate = pretrained_model.config.dropout
        self.hidden_size = pretrained_model.config.n_embd

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, num_classes, bias=False)
        )

        # Initialize classifier weights (following GPT2 paper)
        with torch.no_grad():
            self.classifier[1].weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.pretrained_model(input_ids)  # (batch_size, seq_len, hidden_size)

        pooled_output = hidden_states[:, -1, :]  # last token hidden state

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
