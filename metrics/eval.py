import torch
import torch.nn as nn

def evaluate(best_model, dataset):
    for i, y_true in dataset:
        y_pred = model(i)