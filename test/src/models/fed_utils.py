import torch
import logging

logger = logging.getLogger(__name__)

def federated_average(client_weights, client_sizes):
    """Simple federated averaging"""
    if not client_weights:
        return {}
    
    total_size = sum(client_sizes)
    averaged_weights = {}
    
    # Initialize with first client's structure
    for key in client_weights[0].keys():
        averaged_weights[key] = torch.zeros_like(client_weights[0][key])
    
    # Weighted average
    for client_w, size in zip(client_weights, client_sizes):
        weight = size / total_size
        for key in averaged_weights.keys():
            averaged_weights[key] += weight * client_w[key]
    
    logger.info(f"Averaged weights from {len(client_weights)} clients")
    return averaged_weights
