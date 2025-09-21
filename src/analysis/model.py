import torch
from transformers import DistilBertModel, AutoTokenizer
from torch import nn
import transformers


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = DistilBertModel.from_pretrained(model_name)


class DistilBertForTokenClassification(nn.Module):
    """DistilBert Classifier"""
    def __init__(self, distil_bert: transformers.DistilBertModel, hidden_dim: int, num_pos: int):
        """
        Initlize your classifier with DistilBertModel and a linear layer.
        """
        ############################## START OF YOUR CODE ##############################
        super(DistilBertForTokenClassification, self).__init__()
        self.distil_bert = distil_bert
        self.linear = nn.Linear(hidden_dim, num_pos)
        self.signature = "tune"

        ############################### END OF YOUR CODE ###############################
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Input:
            - input_ids: Tensor with shape (batch size, len), token ids for each input text
            - attention mask: Tensor (batch size, len), mask for padded tokens
        Return:
            - Output: Tensor (batch size, num_pos, len), tag scores for each token
        """
        output = None

        ############################## START OF YOUR CODE ##############################
        outputs = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        outputs = self.linear(last_hidden_state) 
        output = outputs.permute(0, 2, 1) 
        ############################### END OF YOUR CODE ###############################

        return output