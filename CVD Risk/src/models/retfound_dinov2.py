import torch
#neural netwrk module for classification head
import torch.nn as nn
#import the AutoModel from transformers to load the hugging face chcekpoints 
from transformers import AutoModel
# import project abstarct class 
from .base import BaseClassifier, PreprocessingConfig


# create the class 
class RETFoundDINOv2(BaseClassifier):
     def __init__(self, pretrained=True, **kwargs):
        super().__init__()
       
        # load the pretrained model from hugging face
        self.model = AutoModel.from_pretrained("YukunZhou/RETFound_dinov2_meh")
       
