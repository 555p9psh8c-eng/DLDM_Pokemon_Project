"""train_LSTM.py — Step 5: the LSTM model (Lectures 6-7).

The idea (straight from the proposal)
-------------------------------------
The MLP treats every decision in isolation. But a card game is a STORY: what's smart on
turn 8 depends on everything that happened on turns 1-7. So instead of judging each board
alone, we feed the network the WHOLE GAME as a sequence and let a recurrent memory build up
a picture of how the match has developed.

  * Encode each decision's board as a vector (the SAME 280-number state as the MLP).
  * Run those vectors, in order, through an LSTM (Lecture 6). At decision t the LSTM's
    hidden state h_t is a summary of the game *so far* (decisions 1..t).
  * Score each legal option with the SAME conditional-logit head as the MLP — but feed it
    the history-aware h_t instead of the raw board. Softmax over the legal options,
    cross-entropy vs the expert's choice.

So the only thing that changes vs the MLP is the "context": raw board  ->  LSTM memory.
Everything else (per-option features, embeddings, masking, by-game split, top-1/top-3) is
identical, which makes the MLP-vs-LSTM comparison clean.

Why the LSTM is CAUSAL (unidirectional)
---------------------------------------
At decision t of a real game you cannot see the future. A bidirectional LSTM would read
later decisions to predict the current one — information it will NOT have on the live
ladder — so it would cheat in training and fail in deployment. We therefore use a
forward-only LSTM: h_t depends on decisions 1..t only. (The current board x_t is fair game
— it's what you're looking at when you decide.)

Lecture alignment
-----------------
  L6 (RNN/LSTM)   : recurrent hidden state carrying history; LSTM gates fight the
                    vanishing-gradient problem over long games; trained by backprop through
                    time. L7 (seq2seq): we use the encoder half — a many-to-many tagger that
                    emits a decision at every timestep.
  L2              : Adam, dropout, L2 weight decay.
  L3              : backprop through the whole unrolled sequence.

Run it (needs dataset.pt from build_dataset.py):

    ../.venv/bin/python train_LSTM.py
    ../.venv/bin/python train_LSTM.py --lstm-hidden 384 --epochs 25
"""

from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------------
# Per-game dataset: one item = one whole game (a sequence of decisions, each with options)
# --------------------------------------------------------------------------------------
class GameDataset(Dataset):
    def __init__(self, state_n, ranges, offsets, dense_n, opt_card, opt_attack, y, games):
        self.state_n = state_n            # (N, 280) standardized
        self.ranges = ranges              # (n_games, 2): [start, end) decision rows per game
        self.offsets = offsets            # (N+1,) CSR option offsets
        self.dense_n = dense_n            # (T_opts, Fo) standardized
        self.opt_card = opt_card
        self.opt_attack = opt_attack
        self.y = y
        self.games = games                # game ids in this split

    def __len__(self):
        return len(self.games)

    def length_of(self, k):
        g = int(self.games[k]); a, b = self.ranges[g]
        return int(b - a)

    def __getitem__(self, k):
        g = int(self.games[k])
        a, b = int(self.ranges[g][0]), int(self.ranges[g][1])
        rows = range(a, b)                                   # decision rows, chronological
        state_seq = self.state_n[a:b]                        # (T, 280)
        decs = []
        for i in rows:
            o0, o1 = int(self.offsets[i]), int(self.offsets[i + 1])
            decs.append((self.dense_n[o0:o1], self.opt_card[o0:o1],
                         self.opt_attack[o0:o1], int(self.y[i])))
        return state_seq, decs


def collate_games(batch):
    """Pad game sequences to the batch's longest game, and pad every decision's option set
    to the batch's widest menu. Returns two masks: one over timesteps, one over options."""
    B = len(batch)
    lengths = [item[0].size(0) for item in batch]
    Tmax = max(lengths)
    Fs = batch[0][0].size(1)
    state_pad = torch.zeros(B, Tmax, Fs)
    seqmask = torch.zeros(B, Tmax, dtype=torch.bool)

    # flatten decisions in (game, timestep) order — matches H[seqmask] later
    dec_dense, dec_card, dec_attack, ys = [], [], [], []
    for bi, (state_seq, decs) in enumerate(batch):
        T = state_seq.size(0)
        state_pad[bi, :T] = state_seq
        seqmask[bi, :T] = True
        for (d, c, a, yi) in decs:
            dec_dense.append(d); dec_card.append(c); dec_attack.append(a); ys.append(yi)

    D = len(ys)
    Mmax = max(d.size(0) for d in dec_dense)
    Fo = dec_dense[0].size(1)
    dense = torch.zeros(D, Mmax, Fo)
    card = torch.zeros(D, Mmax, dtype=torch.long)
    attack = torch.zeros(D, Mmax, dtype=torch.long)
    optmask = torch.zeros(D, Mmax, dtype=torch.bool)
    y = torch.tensor(ys, dtype=torch.long)
    for di, (d, c, a) in enumerate(zip(dec_dense, dec_card, dec_attack)):
        n = d.size(0)
        dense[di, :n] = d; card[di, :n] = c; attack[di, :n] = a; optmask[di, :n] = True
    return state_pad, seqmask, torch.tensor(lengths), dense, card, attack, optmask, y


class BucketBatcher:
    """Yield batches of games with SIMILAR lengths (minimal padding), shuffling batch order
    each epoch. Lossless — no sequence truncation, even for the rare 1500-decision game."""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.ds = dataset
        self.bs = batch_size
        self.shuffle = shuffle
        order = sorted(range(len(dataset)), key=dataset.length_of)   # by game length
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        batches = self.batches[:]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield [self.ds[k] for k in b]


# --------------------------------------------------------------------------------------
# Model: LSTM encoder + conditional-logit scoring head
# --------------------------------------------------------------------------------------
class LSTMCondLogit(nn.Module):
    def __init__(self, state_dim, opt_dim, card_vocab, attack_vocab,
                 lstm_hidden, head_hidden, dropout, lstm_layers=1,
                 card_emb=16, attack_emb=8):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.card_embed = nn.Embedding(card_vocab + 1, card_emb, padding_idx=0)
        self.attack_embed = nn.Embedding(attack_vocab + 1, attack_emb, padding_idx=0)
        in_dim = lstm_hidden + opt_dim + card_emb + attack_emb
        layers: list[nn.Module] = []
        prev = in_dim
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))            # one SCORE per option
        self.head = nn.Sequential(*layers)

    def forward(self, state_pad, seqmask, dense, card, attack, optmask):
        H, _ = self.lstm(state_pad)                  # (B, Tmax, hidden) — causal
        h = self.drop(H[seqmask])                    # (D, hidden): valid decisions, (game,t) order
        D, M, _ = dense.shape
        h_exp = h[:, None, :].expand(D, M, h.size(1))
        feat = torch.cat([h_exp, dense,
                          self.card_embed(card), self.attack_embed(attack)], dim=-1)
        scores = self.head(feat).squeeze(-1)         # (D, M)
        return scores.masked_fill(~optmask, float("-inf"))


# --------------------------------------------------------------------------------------
# By-game split (returns GAME ids per split, so whole games stay together)
# --------------------------------------------------------------------------------------
def split_games(n_games, seed, fracs=(0.8, 0.1, 0.1)):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_games, generator=g)
    n_tr = int(fracs[0] * n_games)
    n_va = int(fracs[1] * n_games)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


@torch.no_grad()
def evaluate(model, batcher, device):
    model.eval()
    c1 = c3 = total = 0
    loss_sum = 0.0
    for batch in batcher:
        sp, sm, ln, dense, card, attack, om, y = collate_games(batch)
        sp, sm, dense = sp.to(device), sm.to(device), dense.to(device)
        card, attack, om, y = card.to(device), attack.to(device), om.to(device), y.to(device)
        scores = model(sp, sm, dense, card, attack, om)
        loss_sum += F.cross_entropy(scores, y, reduction="sum").item()
        c1 += (scores.argmax(1) == y).sum().item()
        k = min(3, scores.size(1))
        c3 += (scores.topk(k, 1).indices == y[:, None]).any(1).sum().item()
        total += y.numel()
    return c1 / total, c3 / total, loss_sum / total


# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="Train the LSTM (sequence conditional-logit).")
    ap.add_argument("--data", default="dataset.pt")
    ap.add_argument("--out", default="LSTM_baseline.pt")
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--head-hidden", type=int, nargs="+", default=[256])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32, help="games per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    return ap


def train_and_eval(args, save=True, verbose=True):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
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
    n_games = int(game_index.max()) + 1

    # per-game contiguous decision-row ranges (games are stored contiguously & in order)
    change = (torch.nonzero(game_index[1:] != game_index[:-1]).flatten() + 1)
    starts = torch.cat([torch.tensor([0]), change])
    ends = torch.cat([change, torch.tensor([game_index.numel()])])
    ranges = torch.stack([starts, ends], dim=1)               # (n_games, 2)

    g_tr, g_va, g_te = split_games(n_games, args.seed)
    # decision rows per split (for standardization stats + baselines)
    def rows_of(games):
        m = torch.zeros(n_games, dtype=torch.bool); m[games] = True
        return torch.nonzero(m[game_index], as_tuple=True)[0]
    tr_rows = rows_of(g_tr)
    log(f"  games tr/va/te = {len(g_tr)}/{len(g_va)}/{len(g_te)}   decisions train={len(tr_rows)}")

    # standardize state + option features on TRAIN decisions only (no leakage)
    s_mean, s_std = state[tr_rows].mean(0), state[tr_rows].std(0).clamp_min(1e-6)
    state_n = (state - s_mean) / s_std
    oit = torch.zeros(opt_dense.size(0), dtype=torch.bool)
    for i in tr_rows.tolist():
        oit[offsets[i]:offsets[i + 1]] = True
    o_mean, o_std = opt_dense[oit].mean(0), opt_dense[oit].std(0).clamp_min(1e-6)
    dense_n = (opt_dense - o_mean) / o_std

    def batcher_for(games, shuffle):
        ds = GameDataset(state_n, ranges, offsets, dense_n, opt_card, opt_attack, y, games)
        return BucketBatcher(ds, args.batch_size, shuffle=shuffle)

    train_b = batcher_for(g_tr, True)
    val_b = batcher_for(g_va, False)
    test_b = batcher_for(g_te, False)

    te_rows = rows_of(g_te)
    b0 = (y[te_rows] == 0).float().mean().item()
    brand = (1.0 / num_options[te_rows].float()).mean().item()
    log(f"  baselines (test): always-index-0={b0:.3f}   random-legal={brand:.3f}")

    model = LSTMCondLogit(state.size(1), opt_dense.size(1), card_vocab_size, attack_vocab_size,
                          args.lstm_hidden, args.head_hidden, args.dropout,
                          lstm_layers=args.lstm_layers).to(device)
    log(f"Model: LSTM hidden={args.lstm_hidden}x{args.lstm_layers}  head={args.head_hidden}  "
        f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, since = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = seen = 0
        for batch in train_b:
            sp, sm, ln, dense, card, attack, om, yb = collate_games(batch)
            sp, sm, dense = sp.to(device), sm.to(device), dense.to(device)
            card, attack, om, yb = card.to(device), attack.to(device), om.to(device), yb.to(device)
            opt.zero_grad()
            scores = model(sp, sm, dense, card, attack, om)
            loss = F.cross_entropy(scores, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # RNNs: guard exploding grads
            opt.step()
            run_loss += loss.item() * yb.numel(); seen += yb.numel()
        v1, v3, vloss = evaluate(model, val_b, device)
        improved = v1 > best_val
        if improved:
            best_val = v1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        log(f"epoch {epoch:>3}  train_loss={run_loss/seen:.4f}  val_loss={vloss:.4f}  "
            f"val_top1={v1:.4f}  val_top3={v3:.4f}{'  *' if improved else ''}  ({time.time()-t0:.1f}s)")
        if since >= args.patience:
            log(f"Early stop: no val improvement in {args.patience} epochs.")
            break

    model.load_state_dict(best_state)
    t1, t3, tloss = evaluate(model, test_b, device)
    log("\n=== results ===")
    log(f"  best val top1 : {best_val:.4f}")
    log(f"  TEST top1     : {t1:.4f}   top3: {t3:.4f}   (loss {tloss:.4f})")
    log(f"  vs baselines  : always-0={b0:.3f}  random-legal={brand:.3f}  -> +{(t1-b0)*100:.1f} pts")

    if save:
        torch.save({
            "state_dict": best_state, "config": vars(args),
            "state_mean": s_mean, "state_std": s_std, "opt_mean": o_mean, "opt_std": o_std,
            "card_vocab_size": card_vocab_size, "attack_vocab_size": attack_vocab_size,
            "test_acc": t1, "test_top3": t3, "val_acc": best_val,
        }, args.out)
        log(f"  saved -> {args.out}")

    return {"seed": args.seed, "val_top1": best_val, "test_top1": t1, "test_top3": t3,
            "baseline_always0": b0, "baseline_random": brand}


def main():
    train_and_eval(build_parser().parse_args(), save=True, verbose=True)


if __name__ == "__main__":
    main()
