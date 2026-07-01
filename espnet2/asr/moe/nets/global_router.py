import torch
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
import torch.nn.functional as F
import logging
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.swish import Swish

class Linear_Global_Router(torch.nn.Module):
    def __init__(self, input_size, output_size, num_class=3, **kwargs):
        super(Linear_Global_Router, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, masks, audiomasks, **kwargs):
        return (torch.nn.functional.softmax(self.fc(x), dim=-1), None), None

class Attention_Global_Router(torch.nn.Module):
    def __init__(self, input_size, output_size, num_class=3,
                 num_heads=4, linear_units=512, dropout_rate=0.1, 
                 lossweight="0.0,0.5,0.5", class_weight=5.0):
        super(Attention_Global_Router, self).__init__()
        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(input_size)
        self.att = MultiHeadedAttention(num_heads, input_size, dropout_rate)
        self.ffd = PositionwiseFeedForward(input_size, linear_units, dropout_rate)
        self.router = torch.nn.Linear(input_size, output_size, bias=False)
        self.out = torch.nn.Linear(input_size, num_class)
        self.num_class = num_class
        if lossweight == "Weight_None":
            self.weights = None
        else:
            self.weights = [float(x) for x in lossweight.split(",")]
        self.class_weight = class_weight
        logging.info(lossweight)
        logging.info(class_weight)
        
    def forward(self, x, masks, audiomasks, **kwargs):
        residual = x
        x = self.norm1(x)
        x = self.att(x, x, x, masks) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffd(x) + residual
        router = torch.nn.functional.softmax(self.router(x), dim=-1)
        if audiomasks is None:
            return (router, x), None
        classification = self.out(x)
        audiomasks_long = audiomasks.long()
        logits_flat = classification.view(-1, self.num_class)
        targets_flat = audiomasks_long.view(-1)
        loss = self.caculate_loss(logits_flat, targets_flat)
        return (router, x), loss
    
    def caculate_loss(self, logits_flat, targets_flat):
        device = logits_flat.device
        if self.weights is None:
            loss = F.cross_entropy(logits_flat, targets_flat, 
                               ignore_index=0)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, 
                               weight=torch.tensor(self.weights).view(-1).to(device),
                               ignore_index=0)
        return loss * self.class_weight
        
class Attention_Global_Router_woloss(torch.nn.Module):
    def __init__(self, input_size, output_size, num_class=3,
                 num_heads=4, linear_units=512, dropout_rate=0.1, 
                 lossweight="0.0,0.5,0.5", class_weight=5.0):
        super(Attention_Global_Router_woloss, self).__init__()
        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(input_size)
        self.att = MultiHeadedAttention(num_heads, input_size, dropout_rate)
        self.ffd = PositionwiseFeedForward(input_size, linear_units, dropout_rate)
        self.router = torch.nn.Linear(input_size, output_size, bias=False)
        self.num_class = num_class
        if lossweight == "Weight_None":
            self.weights = None
        else:
            self.weights = [float(x) for x in lossweight.split(",")]
        self.class_weight = class_weight
        logging.info(lossweight)
        logging.info(class_weight)
        
    def forward(self, x, masks, audiomasks, **kwargs):
        residual = x
        x = self.norm1(x)
        x = self.att(x, x, x, masks) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffd(x) + residual
        router = torch.nn.functional.softmax(self.router(x), dim=-1)
        return (router, x), None

