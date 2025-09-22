import torch
from transformers import DistilBertModel, AutoTokenizer
from torch import nn
import transformers


class DistilBertForTokenClassification(nn.Module):
    """DistilBert-based classifier for token-level classification tasks.

    Args:
        distil_bert (transformers.DistilBertModel): Pretrained DistilBert model.
        hidden_dim (int): Hidden dimension size from DistilBert output.
        num_pos (int): Number of POS tag classes.

    Attributes:
        distil_bert (transformers.DistilBertModel): Pretrained DistilBert model.
        linear (nn.Linear): Linear layer for classification.
        signature (str): Model signature string.
    """

    def __init__(self, distil_bert: transformers.DistilBertModel, hidden_dim: int, num_pos: int):
        """Initialize the classifier with DistilBert and a linear layer.

        Args:
            distil_bert (transformers.DistilBertModel): Pretrained DistilBert model.
            hidden_dim (int): Hidden dimension size from DistilBert output.
            num_pos (int): Number of POS tag classes.
        """

        super(DistilBertForTokenClassification, self).__init__()
        self.distil_bert = distil_bert
        self.linear = nn.Linear(hidden_dim, num_pos)
        self.signature = "tune"

    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass for token classification.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs with shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Tensor mask for padded tokens with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Tag scores for each token with shape (batch_size, num_pos, seq_len).
        """

        outputs = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        outputs = self.linear(last_hidden_state) 
        output = outputs.permute(0, 2, 1) 

        return output