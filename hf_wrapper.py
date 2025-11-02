import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput


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


class GPTForQuestionAnswering(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.config.n_embd
        self.pad_token_id = pretrained_model.config.pad_token_id
        self.qa_outputs = nn.Linear(self.hidden_size, 2)  # -> (B, L, 2)
        nn.init.normal_(self.qa_outputs.weight, mean=0.0, std=0.02)
        if self.qa_outputs.bias is not None:
            nn.init.zeros_(self.qa_outputs.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        **kwargs
    ):
        # Pass through the LM (ensure we keep last_hidden_state)
        hidden_states = self.pretrained_model(input_ids)
        # hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]  # (B, L, H)

        # Project to 2 logits per token and split
        logits = self.qa_outputs(hidden_states)  # (B,L,2)
        start_logits, end_logits = logits.split(1, dim=-1)  # (B,L,1),(B,L,1)
        start_logits = start_logits.squeeze(-1)  # (B,L)
        end_logits = end_logits.squeeze(-1)  # (B,L)

        # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
        non_pad_mask = (input_ids != self.pad_token_id).to(logits.device, torch.int32)  # (B, L)
        token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)  # (L,)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)  # (B,)
        mask = (token_indices.unsqueeze(0) >= last_non_pad_token.unsqueeze(1))  # (B, L)

        very_neg = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(mask, very_neg)
        end_logits = end_logits.masked_fill(mask, very_neg)

        loss = None
        if start_positions is not None and end_positions is not None:
            # Cross-entropy over sequence length (per token)
            loss_fct = nn.CrossEntropyLoss()
            loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2.0

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=hidden_states,
            attentions=None,
        )
