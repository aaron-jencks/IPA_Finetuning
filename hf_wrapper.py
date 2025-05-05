import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


# class GPTForSequenceClassification(nn.Module):
#     def __init__(self, pretrained_model, num_classes=2):
#         super().__init__()
#         self.pretrained_model = pretrained_model
#         self.num_classes = num_classes
#         self.dropout_rate = pretrained_model.config.dropout
#         self.hidden_size = pretrained_model.config.n_embd
#
#         self.classifier = nn.Sequential(
#             nn.Linear(self.hidden_size, num_classes, bias=False)
#         )
#
#         # Initialize classifier weights (following GPT2 paper)
#         with torch.no_grad():
#             self.classifier[1].weight.data.normal_(mean=0.0, std=0.02)
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         hidden_states = self.pretrained_model(input_ids)  # (batch_size, seq_len, hidden_size)
#
#         pooled_output = hidden_states[:, -1, :]  # last token hidden state
#
#         logits = self.classifier(pooled_output)
#
#         loss = None
#         if labels is not None:
#             loss_fn = nn.CrossEntropyLoss()
#             loss = loss_fn(logits, labels)
#
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=None,
#             attentions=None,
#         )


class GPTForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes=2):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        self.hidden_size = pretrained_model.config.n_embd
        self.pad_token_id = pretrained_model.config.pad_token_id

        self.classifier = nn.Linear(self.hidden_size, num_classes, bias=False)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.pretrained_model(input_ids)  # (batch_size, seq_len, hidden_size)
        logits = self.classifier(hidden_states)

        batch_size, sequence_length = input_ids.shape[:2]

        # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
        non_pad_mask = (input_ids != self.pad_token_id).to(logits.device, torch.int32)
        token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pooled_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=None,
            attentions=None,
        )
