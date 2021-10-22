import numpy as np
import torch

def evaluate_iou(pred, label):
  """ 
  The prediction and gt are binary images. Prediction and gt have values 1 or 0  
  """
  flat_pred = torch.flatten(pred).cpu().detach().numpy()
  flat_label = torch.flatten(label).cpu().detach().numpy()
  intersection = np.sum(np.logical_and(flat_pred,flat_label))
  union = np.sum(np.logical_or(flat_pred,flat_label))
  iou = intersection / union
  return iou