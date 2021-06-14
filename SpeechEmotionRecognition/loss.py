import torch
import torch.nn as nn
import hparams as hp

def SERLoss(predictions, targets):
    return nn.BCEWithLogitsLoss()(input=predictions,target=targets)
    
