"""train_MLP.py — the MLP baseline (Step 4).

This is the model the proposal promises: a deck-agnostic generalization of the competition
repo's conditional-logit behavioural-cloning trainer (github.com/wmh/ptcg-abc,
research/train_bc.py), upgraded from a single linear weight vector on one deck to a
two-layer MLP with ReLU trained on the full dataset.

How it works
------------
For EACH legal option we build a feature vector (type, target, the card/attack it uses).
The MLP scores  f(state, option_i) -> scalar, then softmaxes over the LEGAL options of that
decision. The label is which option the expert chose. This is a neural conditional logit —
exactly train_bc.py's structure, generalized: the network scores WHAT each move does, not
merely its slot in the menu.

Recipe (Lectures 1-2): fully-connected + ReLU, cross-entropy, Adam, dropout + L2.
High-cardinality ids (the hand card, the attack) are learned EMBEDDINGS.

Run it (needs dataset.pt from build_dataset.py):

    ../.venv/bin/python train_MLP.py
    ../.venv/bin/python train_MLP.py --epochs 40 --hidden 512 256

Success = clearly beating the always-index-0 baseline (~20%), while being a model you can
actually explain ("it scored this action high because ...").
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------------------
# Ragged (CSR) dataset: one item = one decision with its variable-length option set
# --------------------------------------------------------------------------------------
class DecisionDataset(Dataset):
    def __init__(self, state, offsets, opt_dense, opt_card, opt_attack, y, rows):
        # `rows` = the decision indices belonging to this split.
        self.state = state
        self.offsets = offsets
        self.opt_dense = opt_dense
        self.opt_card = opt_card
        self.opt_attack = opt_attack
        self.y = y
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = int(self.rows[k])
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return (self.state[i], self.opt_dense[a:b], self.opt_card[a:b],
                self.opt_attack[a:b], int(self.y[i]))


def collate(batch):
    """Pad the variable option counts to the batch max; return a validity mask."""
    B = len(batch)
    M = max(item[1].size(0) for item in batch)
    Fs = batch[0][0].size(0)
    Fo = batch[0][1].size(1)
    state = torch.zeros(B, Fs)
    dense = torch.zeros(B, M, Fo)
    card = torch.zeros(B, M, dtype=torch.long)
    attack = torch.zeros(B, M, dtype=torch.long)
    mask = torch.zeros(B, M, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.long)
    for bi, (s, d, c, a, yi) in enumerate(batch):
        n = d.size(0)
        state[bi] = s
        dense[bi, :n] = d
        card[bi, :n] = c
        attack[bi, :n] = a
        mask[bi, :n] = True
        y[bi] = yi
    return state, dense, card, attack, mask, y


# --------------------------------------------------------------------------------------
# Model: neural conditional logit
# --------------------------------------------------------------------------------------
class CondLogitMLP(nn.Module):
    def __init__(self, state_dim, opt_dim, card_vocab, attack_vocab,
                 hidden, dropout, card_emb=16, attack_emb=8):
        super().__init__()
        self.card_embed = nn.Embedding(card_vocab + 1, card_emb, padding_idx=0)
        self.attack_embed = nn.Embedding(attack_vocab + 1, attack_emb, padding_idx=0)
        in_dim = state_dim + opt_dim + card_emb + attack_emb
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))       # one SCORE per option
        self.scorer = nn.Sequential(*layers)

    def forward(self, state, dense, card, attack, mask):
        B, M, _ = dense.shape
        state_exp = state[:, None, :].expand(B, M, state.size(1))
        feat = torch.cat([state_exp, dense,
                          self.card_embed(card), self.attack_embed(attack)], dim=-1)
        scores = self.scorer(feat).squeeze(-1)               # (B, M)
        return scores.masked_fill(~mask, float("-inf"))


# --------------------------------------------------------------------------------------
# By-game split (identical policy to train_mlp.py)
# --------------------------------------------------------------------------------------
def split_by_game(game_index, seed, fracs=(0.8, 0.1, 0.1)):
    n_games = int(game_index.max()) + 1
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_games, generator=g)
    n_tr = int(fracs[0] * n_games)
    n_va = int(fracs[1] * n_games)
    sets = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    out = {}
    for name, games in sets.items():
        m = torch.zeros(n_games, dtype=torch.bool)
        m[games] = True
        out[name] = torch.nonzero(m[game_index], as_tuple=True)[0]   # decision row ids
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (top1_acc, top3_acc, mean_loss). Top-3 = 'was the expert's move among the
    model's 3 highest-scored options?' — a fairer read when several moves are reasonable."""
    model.eval()
    correct1 = correct3 = total = 0
    loss_sum = 0.0
    for state, dense, card, attack, mask, y in loader:
        state, dense = state.to(device), dense.to(device)
        card, attack, mask, y = card.to(device), attack.to(device), mask.to(device), y.to(device)
        scores = model(state, dense, card, attack, mask)
        loss_sum += F.cross_entropy(scores, y, reduction="sum").item()
        correct1 += (scores.argmax(1) == y).sum().item()
        k = min(3, scores.size(1))
        top = scores.topk(k, dim=1).indices              # (B, k)
        correct3 += (top == y[:, None]).any(1).sum().item()
        total += y.numel()
    return correct1 / total, correct3 / total, loss_sum / total


def build_parser():
    ap = argparse.ArgumentParser(description="Train the conditional-logit MLP baseline.")
    ap.add_argument("--data", default="dataset.pt")
    ap.add_argument("--out", default="MLP_baseline.pt")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    return ap


def train_and_eval(args, save=True, verbose=True):
    """Train once with the given args and return a metrics dict. Reusable by run_seeds.py."""
    torch.manual_seed(args.seed)
    log = print if verbose else (lambda *a, **k: None)
    device = "cpu"
    if not args.cpu:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    log(f"Device: {device}")

    log(f"Loading {args.data} ...")
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    state, offsets = d["state"], d["offsets"]
    opt_dense, opt_card, opt_attack = d["opt_dense"], d["opt_card"], d["opt_attack"]
    y, num_options, game_index = d["y"], d["num_options"], d["game_index"]
    card_vocab_size = d["meta"]["card_vocab_size"]
    attack_vocab_size = d["meta"]["attack_vocab_size"]
    log(f"  decisions={state.size(0)}  options={opt_dense.size(0)}  "
        f"state_dim={state.size(1)}  opt_dim={opt_dense.size(1)}  "
        f"cards={card_vocab_size}  attacks={attack_vocab_size}")

    splits = split_by_game(game_index, args.seed)
    log(f"  rows  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    # --- standardize state + dense using TRAIN stats only (no leakage) ---
    tr_rows = splits["train"]
    s_mean = state[tr_rows].mean(0)
    s_std = state[tr_rows].std(0).clamp_min(1e-6)
    state_n = (state - s_mean) / s_std
    # option rows that belong to train decisions
    opt_is_train = torch.zeros(opt_dense.size(0), dtype=torch.bool)
    for i in tr_rows.tolist():
        opt_is_train[offsets[i]:offsets[i + 1]] = True
    o_mean = opt_dense[opt_is_train].mean(0)
    o_std = opt_dense[opt_is_train].std(0).clamp_min(1e-6)
    dense_n = (opt_dense - o_mean) / o_std

    def loader_for(name, shuffle):
        ds = DecisionDataset(state_n, offsets, dense_n, opt_card, opt_attack, y, splits[name])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader = loader_for("train", True)
    val_loader = loader_for("val", False)
    test_loader = loader_for("test", False)

    # --- baselines on the test split ---
    yte, note = y[splits["test"]], num_options[splits["test"]]
    b0 = (yte == 0).float().mean().item()
    brand = (1.0 / note.float()).mean().item()
    log(f"  baselines (test): always-index-0={b0:.3f}   random-legal={brand:.3f}")

    model = CondLogitMLP(state.size(1), opt_dense.size(1), card_vocab_size,
                         attack_vocab_size, args.hidden, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: hidden={args.hidden}  params={n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, since = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = seen = 0
        for state_b, dense_b, card_b, attack_b, mask_b, y_b in train_loader:
            state_b, dense_b = state_b.to(device), dense_b.to(device)
            card_b, attack_b = card_b.to(device), attack_b.to(device)
            mask_b, y_b = mask_b.to(device), y_b.to(device)
            opt.zero_grad()
            scores = model(state_b, dense_b, card_b, attack_b, mask_b)
            loss = F.cross_entropy(scores, y_b)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y_b.numel()
            seen += y_b.numel()
        val_top1, val_top3, val_loss = evaluate(model, val_loader, device)
        improved = val_top1 > best_val
        if improved:
            best_val = val_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        log(f"epoch {epoch:>3}  train_loss={run_loss/seen:.4f}  val_loss={val_loss:.4f}  "
            f"val_top1={val_top1:.4f}  val_top3={val_top3:.4f}"
            f"{'  *' if improved else ''}  ({time.time()-t0:.1f}s)")
        if since >= args.patience:
            log(f"Early stop: no val improvement in {args.patience} epochs.")
            break

    model.load_state_dict(best_state)
    test_top1, test_top3, test_loss = evaluate(model, test_loader, device)
    log("\n=== results ===")
    log(f"  best val top1 : {best_val:.4f}")
    log(f"  TEST top1     : {test_top1:.4f}   top3: {test_top3:.4f}   (loss {test_loss:.4f})")
    log(f"  vs baselines  : always-0={b0:.3f}  random-legal={brand:.3f}  "
        f"-> +{(test_top1-b0)*100:.1f} pts over always-0")

    if save:
        torch.save({
            "state_dict": best_state,
            "config": vars(args),
            "state_mean": s_mean, "state_std": s_std,
            "opt_mean": o_mean, "opt_std": o_std,
            "card_vocab_size": card_vocab_size, "attack_vocab_size": attack_vocab_size,
            "test_acc": test_top1, "test_top3": test_top3, "val_acc": best_val,
        }, args.out)
        log(f"  saved -> {args.out}")

    return {"seed": args.seed, "val_top1": best_val,
            "test_top1": test_top1, "test_top3": test_top3,
            "baseline_always0": b0, "baseline_random": brand}


def main():
    args = build_parser().parse_args()
    train_and_eval(args, save=True, verbose=True)


if __name__ == "__main__":
    main()
