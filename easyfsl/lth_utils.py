import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

mask = []
step = 0

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        t = p.data.cpu().numpy()
        nz = np.count_nonzero(t)
        tot = t.size
        nonzero += nz
        total += tot
    return (round((nonzero/total)*100, 1))


# def make_mask(model):
#     mask = []
#     for name, p in model.named_parameters():
#         if 'weight' in name:
#             mask.append(np.ones_like(p.data.cpu().numpy()))
#     return mask

def make_mask(model):
    """
    Create a mask list matching only weight tensors with dim > 1
    (Conv and Linear layers, skipping BN 1-D weights).
    """
    mask = []
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() > 1:
            mask.append(np.ones(p.shape, dtype=np.float32))
    return mask


# def prune_by_percentile(percent, model):
#     global mask
#     idx = 0
#     for name, p in model.named_parameters():
#         if 'weight' not in name:
#             continue

#         w = p.data.cpu().numpy()
#         alive = w[np.nonzero(w)]
#         if alive.size == 0:
#             idx += 1
#             continue

#         cutoff = np.percentile(abs(alive), percent)
#         new_mask = np.where(abs(w) < cutoff, 0, mask[idx])

#         p.data = torch.from_numpy(w * new_mask).to(p.device)
#         mask[idx] = new_mask
#         idx += 1
def prune_by_percentile(percent, model, mask_list):
    """
    Zeroes out weights in the backbone (model) based on global percentile threshold.
    Updates mask_list in place.
    """
    # collect all surviving weights
    all_weights = []
    for (name, p), m in zip(model.named_parameters(), mask_list):
        if "weight" in name and p.dim() > 1:
            w = p.detach().cpu().numpy()
            all_weights.extend(np.abs(w[m == 1]))

    if len(all_weights) == 0:
        print("Warning: No weights available for pruning (mask empty).")
        return

    threshold = np.percentile(all_weights, percent)

    # prune
    idx = 0
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() > 1:
            w = p.detach().cpu().numpy()
            new_mask = (np.abs(w) > threshold).astype(np.float32)
            mask_list[idx] = new_mask   # update mask
            idx += 1



# def original_initialization(model, initial_state):
#     global mask
#     idx = 0
#     for name, p in model.named_parameters():
#         w = initial_state[name].cpu().numpy()
#         if "weight" in name:
#             p.data = torch.from_numpy(w * mask[idx]).to(p.device)
#             idx += 1
#         else:
#             p.data = initial_state[name].to(p.device)

def original_initialization(model, mask_list, initial_state_dict):
    """
    Restore only surviving weights from initial weights.
    """
    idx = 0
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() > 1:
            init_w = initial_state_dict[name].cpu().numpy()
            m = mask_list[idx]
            restored_w = init_w * m + p.detach().cpu().numpy() * (1 - m)
            p.data = torch.tensor(restored_w, device=p.device, dtype=p.dtype)
            idx += 1



def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.zeros_(m.bias)
