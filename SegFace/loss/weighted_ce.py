import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from torch import Tensor

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, seg_output, seg_labels, edge_map):
        """
        Calculate weighted cross-entropy loss.

        Parameters:
        seg_output (Tensor): The raw, unnormalized scores for each class. Shape (N, C, H, W).
        seg_labels (Tensor): The ground truth labels. Shape (N, H, W).
        edge_map (Tensor): The edge map where edges are marked with ones and non-edges with zeros. Shape (N, H, W).

        Returns:
        Tensor: The scalar loss.
        """

        # Ensure the edge_map is a floating point tensor for proper weight application
        edge_map = edge_map.float()
        
        # Compute the log softmax of the input logits
        log_probabilities = F.log_softmax(seg_output, dim=1)
        
        # Gather the log probabilities using the target labels
        log_probabilities = log_probabilities.gather(1, seg_labels.unsqueeze(1)).squeeze(1)
        
        # Compute the weighted loss
        weighted_loss = -log_probabilities * edge_map
        
        # Take the mean of the weighted loss, but only for the edges as per the edge_map
        # Avoid division by zero by only considering non-zero elements in edge_map
        loss = weighted_loss[edge_map > 0].mean()

        return loss