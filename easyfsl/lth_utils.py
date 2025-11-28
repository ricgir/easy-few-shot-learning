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


def make_mask(model):
    mask = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            mask.append(np.ones_like(p.data.cpu().numpy()))
    return mask


def prune_by_percentile(percent, model):
    global mask
    idx = 0
    for name, p in model.named_parameters():
        if 'weight' not in name:
            continue

        w = p.data.cpu().numpy()
        alive = w[np.nonzero(w)]
        if alive.size == 0:
            idx += 1
            continue

        cutoff = np.percentile(abs(alive), percent)
        new_mask = np.where(abs(w) < cutoff, 0, mask[idx])

        p.data = torch.from_numpy(w * new_mask).to(p.device)
        mask[idx] = new_mask
        idx += 1


def original_initialization(model, initial_state):
    global mask
    idx = 0
    for name, p in model.named_parameters():
        w = initial_state[name].cpu().numpy()
        if "weight" in name:
            p.data = torch.from_numpy(w * mask[idx]).to(p.device)
            idx += 1
        else:
            p.data = initial_state[name].to(p.device)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.zeros_(m.bias)
