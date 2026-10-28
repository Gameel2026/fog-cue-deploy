import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

import fd_config as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class CueLSTM(nn.Module):
    def __init__(self, n_in=72, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def standardisation(F, idx):
    mu = F[idx].mean(0); sd = F[idx].std(0); sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def standardised(F, mu, sd):
    return np.clip((F - mu) / sd, -10, 10).astype(np.float32)


def batch(Zs, P, rows):
    from fd_data import gather
    return torch.from_numpy(gather(Zs, P[rows]))


@torch.no_grad()
def prob(model, Zs, P, rows):
    model.eval(); out = []
    for k in range(0, len(rows), 4096):
        out.append(torch.softmax(model(batch(Zs, P, rows[k:k + 4096]).to(DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def train(Zs, P, y, idx, epochs, cue_factor, seed=C.SEED, val_idx=None):
    set_seed(seed)
    model = CueLSTM(Zs.shape[1]).to(DEVICE)
    cnt = np.bincount(y[idx], minlength=2).astype(float)
    w = len(idx) / (2 * np.maximum(cnt, 1)); w[1] *= cue_factor
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    idx = np.asarray(idx); yt = torch.from_numpy(y.astype(np.int64))
    best, best_ep, wait, best_state = -1.0, epochs, 0, None
    for ep in range(1, epochs + 1):
        model.train(); perm = idx[torch.randperm(len(idx)).numpy()]
        for k in range(0, len(perm), C.BATCH):
            b = perm[k:k + C.BATCH]; xb, yb = batch(Zs, P, b).to(DEVICE), yt[torch.from_numpy(b)].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb + 0.05 * torch.randn_like(xb)), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if val_idx is not None:
            bacc = balanced_accuracy_score(y[val_idx], (prob(model, Zs, P, np.asarray(val_idx)) >= 0.5).astype(int))
            if bacc > best:
                best, best_ep, wait = bacc, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= C.PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


def export(model, mu, sd, theta, n_past, path_prefix):
    from fd_data import filter_design
    st = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    sos, zi = filter_design()
    base = {"mu": np.asarray(mu, np.float64), "sd": np.asarray(sd, np.float64), "sos": sos, "zi_unit": zi,
            "theta_on": float(theta[0]), "theta_off": float(theta[1]), "n_past": int(n_past),
            "fs": float(C.FS), "win": int(C.WIN), "nfft": int(C.NFFT), "mode_res": float(C.MODE_RES)}
    w = {f"{k}_l{l}": st[f"lstm.{k}_l{l}"].astype(np.float32) for l in range(2) for k in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]}
    w["head_weight"] = st["head.1.weight"].astype(np.float32); w["head_bias"] = st["head.1.bias"].astype(np.float32)
    np.savez(path_prefix + "_fp32.npz", **base, **w)
    q = dict(w)
    for k in list(w):
        if "weight" in k:
            s = np.maximum(np.abs(w[k]).max(1), 1e-12) / 127.0
            q[k] = np.clip(np.round(w[k] / s[:, None]), -127, 127).astype(np.int8); q[k + "__scale"] = s.astype(np.float32)
    np.savez(path_prefix + "_int8.npz", **base, **q)
    torch.save(model.state_dict(), path_prefix + "_state.pt")


def load(prefix):
    d = np.load(prefix + "_fp32.npz")
    model = CueLSTM(len(d["mu"])).to(DEVICE)
    model.load_state_dict(torch.load(prefix + "_state.pt", map_location=DEVICE))
    model.eval()
    return model, d["mu"].astype(np.float64), d["sd"].astype(np.float64), (float(d["theta_on"]), float(d["theta_off"])), int(d["n_past"])
