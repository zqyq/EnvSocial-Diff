from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

class BERTEncoder(nn.Module):
    def __init__(self, model_path=None, device='cpu'):
        super(BERTEncoder, self).__init__()
        self.device = device

        # 加载 tokenizer 和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path).to(device)
        self.bert.eval()  # 设置为评估模式

    def forward(self, text: str):
        # 编码输入
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量, shape: [1, 768]

        # return cls_embedding.squeeze(0)
        return cls_embedding