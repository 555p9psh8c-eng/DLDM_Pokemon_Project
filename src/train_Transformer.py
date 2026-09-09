"""train_Transformer.py — Step 6: the Transformer model (Lecture 8).

The idea (straight from the proposal)
-------------------------------------
The LSTM carries the game's history in ONE hidden vector that gets rewritten every
timestep — turn 3's detail has to survive 40 rewrites to still matter on turn 43. The
Transformer drops recurrence entirely: at decision t it ATTENDS directly to every earlier
decision, so "I benched a second Charmander on turn 3" is one hop away, not forty.

  * Encode each decision's board as a vector (the SAME 280-number state as MLP/LSTM).
  * Run those vectors through a stack of causal self-attention blocks (Lecture 8). At
    decision t the output h_t is a history-aware summary built by *looking directly* at
    decisions 1..t and weighting whichever ones matter.
  * Score each legal option with the SAME conditional-logit head as MLP/LSTM — fed the
    attention-built h_t. Softmax over the legal options, cross-entropy vs the expert.

So again the ONLY thing that changes is the "context": raw board -> LSTM memory ->
attention over the whole past. Per-option features, embeddings, masking, by-game split and
top-1/top-3 are identical across all three models, which keeps the comparison clean.

Why the attention is CAUSAL (masked)
------------------------------------
Same argument as the LSTM. At decision t of a real game you cannot see the future, so
self-attention must not look at t+1, t+2, ... A plain (bidirectional) encoder would read
later decisions to predict the current one — information it will NOT have on the live
ladder — inflating test accuracy and then collapsing in deployment. We therefore apply the
triangular mask from L8: e_{l,t} = -inf whenever t > l. `test_transformer.py` asserts this
numerically rather than trusting the comment.

Lecture alignment (L8) — every piece is built here, not imported from nn.Transformer*
-------------------------------------------------------------------------------------
  Positional encoding  : sin/cos frequency code (slide 10), added to the projected input
                         (slide 12). `--learned-pos` switches to the learned table (slide 11).
  Multi-head attention : per-head q/k/v, scaled dot-product, softmax, concat (slide 15).
  Nonlinearity         : position-wise 2-layer feedforward with ReLU (slides 18, 28) —
                         without it a stack of attention layers stays linear (slide 17).
  Residual + LayerNorm : "Add & Norm" around both sublayers (slide 28). Post-norm by
                         default to match the slide; `--pre-norm` for the modern variant.
  Masked attention     : causal triangular mask (slide 21).
  L2                   : Adam, dropout, weight decay.

Run it (needs dataset.pt from build_dataset.py):

    .venv/bin/python train_Transformer.py
    .venv/bin/python train_Transformer.py --d-model 256 --n-layers 4 --n-heads 8

Test it (no dataset needed — builds its own synthetic one):

    .venv/bin/python test_transformer.py
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------------
# Per-game dataset: one item = one whole game (a sequence of decisions, each with options)
# Identical to train_LSTM.py — the two models must see byte-for-byte the same data.
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
    each epoch. Lossless — no sequence truncation, even for the rare 1500-decision game.

    Extra vs the LSTM's batcher: attention costs O(T^2) memory, not O(T). A bucket of 32
    games x 1500 decisions would allocate ~1500^2 x 32 x heads attention scores and blow up
    RAM. So we also cap each batch by an ATTENTION BUDGET (sum of T^2): long-game buckets
    automatically get fewer games per batch. Still lossless — nothing is truncated, the
    batches just get smaller exactly where the quadratic cost bites.
    """
    def __init__(self, dataset, batch_size, attn_budget=4_000_000, shuffle=True):
        self.ds = dataset
        self.shuffle = shuffle
        order = sorted(range(len(dataset)), key=dataset.length_of)   # by game length
        self.batches = []
        cur: list[int] = []
        for k in order:
            t = dataset.length_of(k)
            # would adding this game exceed either cap? then close the current batch first
            if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * t * t > attn_budget):
                self.batches.append(cur)
                cur = []
            cur.append(k)
        if cur:
            self.batches.append(cur)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        batches = self.batches[:]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield [self.ds[k] for k in b]


# --------------------------------------------------------------------------------------
# L8 building blocks — written out rather than pulled from torch.nn.Transformer*
# --------------------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """p_t from slide 10: alternating sin/cos over geometrically spaced frequencies.

        p_t[2i]   = sin(t / 10000^(2i/d))
        p_t[2i+1] = cos(t / 10000^(2i/d))

    Computed on the fly, so it extends to any game length (our longest games run to ~1500
    decisions) — the property the learned table on slide 11 does not have.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, T, device, dtype):
        # Build at the working precision: rounding sin/cos through float32 and casting up
        # would cost ~1e-7 of accuracy, which is invisible in training but shows up the
        # moment you check the formula in double (see test_transformer.py test 1).
        calc = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        pos = torch.arange(T, device=device, dtype=calc)[:, None]           # (T, 1)
        i = torch.arange(0, self.d_model, 2, device=device, dtype=calc)     # (d/2,)
        freq = torch.exp(-math.log(10000.0) * i / self.d_model)             # 1/10000^(2i/d)
        pe = torch.zeros(T, self.d_model, device=device, dtype=calc)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)[:, : pe[:, 1::2].size(1)]   # odd d_model safe
        return pe.to(dtype)


class LearnedPositionalEncoding(nn.Module):
    """Slide 11: just learn P = [p_1..p_T]. More flexible, but capped at max_len."""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.table = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, T, device, dtype):
        if T > self.max_len:
            raise ValueError(
                f"Game has {T} decisions but --pos-max-len is {self.max_len}. Raise it, or "
                f"use the default sinusoidal encoding which has no length limit.")
        return self.table(torch.arange(T, device=device)).to(dtype)


class MultiHeadSelfAttention(nn.Module):
    """Slide 15, with the causal mask of slide 21.

    Per head i:  q = W_q h,  k = W_k h,  v = W_v h
                 e_{l,t} = q_l . k_t / sqrt(d_head)      (-inf if t > l, or t is padding)
                 alpha   = softmax_t(e_{l,t})
                 a_{l,i} = sum_t alpha_{l,t,i} v_{t,i}
    then the heads are concatenated and mixed by W_o.

    The 1/sqrt(d_head) scaling is the standard fix for dot products growing with dimension
    (Vaswani et al.) — without it the softmax saturates and gradients vanish.
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _heads(self, x, B, T):
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)   # (B, H, T, d_head)

    def forward(self, x, attn_bias):
        """x: (B, T, d_model).  attn_bias: (B, 1, T, T) additive, 0 = allowed, -inf = blocked."""
        B, T, _ = x.shape
        q = self._heads(self.W_q(x), B, T)
        k = self._heads(self.W_k(x), B, T)
        v = self._heads(self.W_v(x), B, T)

        e = (q @ k.transpose(-2, -1)) * self.scale        # (B, H, T, T)
        e = e + attn_bias                                  # causal + padding mask
        alpha = self.attn_drop(e.softmax(dim=-1))
        a = alpha @ v                                      # (B, H, T, d_head)
        a = a.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)   # concat heads
        return self.resid_drop(self.W_o(a))


class TransformerBlock(nn.Module):
    """One encoder layer: masked multi-head self-attention, then a position-wise ReLU
    feedforward, each wrapped in a residual connection + LayerNorm (slide 28).

    post_norm=True  (default, matches slide 28 / Vaswani):  x <- LN(x + sublayer(x))
    post_norm=False (`--pre-norm`, the modern variant):     x <- x + sublayer(LN(x))
    Pre-norm trains more stably when you stack many layers; post-norm is what the lecture
    draws, so it is the default here.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, post_norm=True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(                      # h = W2 ReLU(W1 a + b1) + b2
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.post_norm = post_norm

    def forward(self, x, attn_bias):
        if self.post_norm:
            x = self.norm1(x + self.attn(x, attn_bias))
            x = self.norm2(x + self.ff(x))
        else:
            x = x + self.attn(self.norm1(x), attn_bias)
            x = x + self.ff(self.norm2(x))
        return x


def causal_padding_bias(seqmask, dtype):
    """Build the additive attention mask: (B, 1, T, T), 0 where allowed, -inf where not.

    Two things are blocked:
      * the FUTURE (slide 21): query l may not attend to key t > l;
      * PADDING: keys past a game's real length are not part of that game.

    No row can end up fully -inf: the causal mask always leaves the diagonal open, and
    t=0 is a real decision in every game, so every query keeps at least one live key.
    """
    B, T = seqmask.shape
    device = seqmask.device
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    blocked = future[None, None] | (~seqmask)[:, None, None, :]      # (B, 1, T, T)
    return torch.zeros(B, 1, T, T, dtype=dtype, device=device).masked_fill_(
        blocked, torch.finfo(dtype).min)


# --------------------------------------------------------------------------------------
# Model: causal Transformer encoder + the same conditional-logit scoring head
# --------------------------------------------------------------------------------------
class TransformerCondLogit(nn.Module):
    def __init__(self, state_dim, opt_dim, card_vocab, attack_vocab,
                 d_model=192, n_heads=4, n_layers=2, d_ff=384, dropout=0.1,
                 head_hidden=(256,), card_emb=16, attack_emb=8,
                 learned_pos=False, pos_max_len=2048, post_norm=True):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)      # emb(x_t) — slide 12
        self.pos = (LearnedPositionalEncoding(d_model, pos_max_len) if learned_pos
                    else SinusoidalPositionalEncoding(d_model))
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, post_norm=post_norm)
            for _ in range(n_layers)
        ])
        # Pre-norm leaves the stack's output unnormalized; normalize once before the head.
        self.final_norm = nn.Identity() if post_norm else nn.LayerNorm(d_model)

        # --- scoring head: identical in shape to the MLP's and the LSTM's ---
        self.card_embed = nn.Embedding(card_vocab + 1, card_emb, padding_idx=0)
        self.attack_embed = nn.Embedding(attack_vocab + 1, attack_emb, padding_idx=0)
        in_dim = d_model + opt_dim + card_emb + attack_emb
        layers: list[nn.Module] = []
        prev = in_dim
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))            # one SCORE per option
        self.head = nn.Sequential(*layers)

    def encode(self, state_pad, seqmask):
        """(B, T, state_dim) -> (B, T, d_model), each position summarizing decisions 1..t."""
        B, T, _ = state_pad.shape
        x = self.input_proj(state_pad)
        x = x + self.pos(T, x.device, x.dtype)[None]          # slide 12: emb(x_t) + p_t
        x = self.input_drop(x)
        bias = causal_padding_bias(seqmask, x.dtype)
        for block in self.blocks:
            x = block(x, bias)
        return self.final_norm(x)

    def forward(self, state_pad, seqmask, dense, card, attack, optmask):
        H = self.encode(state_pad, seqmask)          # (B, T, d_model) — causal
        h = H[seqmask]                               # (D, d_model): valid decisions, (game,t) order
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
    ap = argparse.ArgumentParser(description="Train the Transformer (sequence conditional-logit).")
    ap.add_argument("--data", default="dataset.pt")
    ap.add_argument("--out", default="Transformer_baseline.pt")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=384)
    ap.add_argument("--head-hidden", type=int, nargs="+", default=[256])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--learned-pos", action="store_true",
                    help="learned positional table (slide 11) instead of sin/cos (slide 10)")
    ap.add_argument("--pos-max-len", type=int, default=2048, help="only used with --learned-pos")
    ap.add_argument("--pre-norm", action="store_true",
                    help="x + sublayer(LN(x)) instead of the lecture's LN(x + sublayer(x))")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200,
                    help="linear LR warmup steps; post-norm transformers need it to start stably")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32, help="games per batch (upper bound)")
    ap.add_argument("--attn-budget", type=int, default=4_000_000,
                    help="cap on sum(T^2) per batch; shrinks batches for very long games")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="val_top1 must rise by at least this much to count as improvement; "
                         "smaller gains don't reset patience, so a plateau stops early")
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
        return BucketBatcher(ds, args.batch_size, attn_budget=args.attn_budget, shuffle=shuffle)

    train_b = batcher_for(g_tr, True)
    val_b = batcher_for(g_va, False)
    test_b = batcher_for(g_te, False)

    te_rows = rows_of(g_te)
    b0 = (y[te_rows] == 0).float().mean().item()
    brand = (1.0 / num_options[te_rows].float()).mean().item()
    log(f"  baselines (test): always-index-0={b0:.3f}   random-legal={brand:.3f}")

    model = TransformerCondLogit(
        state.size(1), opt_dense.size(1), card_vocab_size, attack_vocab_size,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, head_hidden=tuple(args.head_hidden),
        learned_pos=args.learned_pos, pos_max_len=args.pos_max_len,
        post_norm=not args.pre_norm).to(device)
    log(f"Model: d_model={args.d_model} heads={args.n_heads} layers={args.n_layers} "
        f"d_ff={args.d_ff} {'pre' if args.pre_norm else 'post'}-norm "
        f"pos={'learned' if args.learned_pos else 'sin/cos'}  head={args.head_hidden}  "
        f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    best_val, best_state, since = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = seen = 0
        for batch in train_b:
            sp, sm, ln, dense, card, attack, om, yb = collate_games(batch)
            sp, sm, dense = sp.to(device), sm.to(device), dense.to(device)
            card, attack, om, yb = card.to(device), attack.to(device), om.to(device), yb.to(device)
            step += 1
            if args.warmup > 0 and step <= args.warmup:      # linear LR warmup
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * step / args.warmup
            opt.zero_grad()
            scores = model(sp, sm, dense, card, attack, om)
            loss = F.cross_entropy(scores, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            run_loss += loss.item() * yb.numel(); seen += yb.numel()
        v1, v3, vloss = evaluate(model, val_b, device)
        improved = v1 > best_val + args.min_delta      # only a MEANINGFUL gain resets patience
        if improved:
            best_val = v1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
        log(f"epoch {epoch:>3}  train_loss={run_loss/seen:.4f}  val_loss={vloss:.4f}  "
            f"val_top1={v1:.4f}  val_top3={v3:.4f}{'  *' if improved else ''}  ({time.time()-t0:.1f}s)")
        if since >= args.patience:
            log(f"Early stop: val_top1 gained < {args.min_delta} for {args.patience} "
                f"straight epochs (best {best_val:.4f}).")
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
