"""train_Transformer_improved.py — Improved Transformer with deeper architecture.

Current Performance Analysis (seed average)
-------------------------------------------
  LSTM:        65.7% test accuracy (256 hidden, 1 layer)
  Transformer: 66.6% test accuracy (d_model=192, n_layers=2, n_heads=4)
  Gap:         Only 0.9% improvement despite higher capacity

Root Cause Analysis
-------------------
The current Transformer is SHALLOW and NARROW:
  * 2 layers: Modern transformers use 6-12+ layers. With only 2 layers, attention
    can only look "2 hops back" in the computation graph. For a 60-decision game,
    that's insufficient to capture long-range dependencies.
  * d_model=192: Small representation space limits what the model can learn.
  * 4 heads: With d_head=48, each head has limited capacity for different attention
    patterns (e.g., "recent moves", "board state", "opponent strategy").
  * d_ff=384: Only 2x expansion in feedforward — modern practice uses 4x.

The result: the Transformer can't leverage its architectural advantages. It's like
running a race car in first gear — technically superior, but artificially limited.

Improvements in This Version
-----------------------------
1. DEEPER ARCHITECTURE (4-6 layers):
   - More layers = more computation steps = better long-range modeling
   - Each layer refines the representation with a new round of attention
   - Target: 6 layers (matching BERT-base, GPT-2 small)

2. WIDER MODEL (d_model=256-512):
   - More dimensions = richer representations
   - More room for different features (card types, HP, energy, strategy)
   - Target: d_model=384 (sweet spot for this task)

3. MORE HEADS (8-16):
   - Each head can specialize: recent history, early game, card synergies, etc.
   - With 8 heads and d_model=384, each head gets d_head=48 (same as before)
   - Better than 4 heads × 48 because of diversity

4. BETTER FEEDFORWARD (4x expansion):
   - d_ff = 4 × d_model (standard modern practice)
   - More nonlinear capacity to transform the attention outputs

5. PRE-NORM ARCHITECTURE:
   - x + sublayer(LN(x)) instead of LN(x + sublayer(x))
   - Trains more stably for deep networks (GPT-2/3, BERT all use pre-norm variants)

6. IMPROVED LEARNING RATE SCHEDULE:
   - Longer warmup (1000 steps vs 200) for stability
   - Cosine decay after warmup for better convergence
   - Lower peak LR (1e-4 vs 3e-4) with weight decay for regularization

7. DROPOUT SCHEDULE:
   - Start with higher dropout (0.2) to prevent early overfitting
   - Optional: reduce dropout in later epochs (not implemented yet)

8. GRADIENT CLIPPING:
   - Kept at 1.0 but monitored more carefully
   - Transformers can have gradient spikes, especially early in training

9. LONGER TRAINING:
   - 40 epochs (vs 25) with patience=8 (vs 5)
   - Transformers are slower to converge than LSTMs
   - Early stopping still prevents actual overfitting

10. ATTENTION PATTERN ANALYSIS (optional):
    - Save attention weights for key batches
    - Visualize which decisions attend to which history
    - Debug mode: --save-attention

Expected Performance
--------------------
With these changes, targeting 72-75% test accuracy:
  * +6-9 points over current 66.6%
  * +11-14 points over LSTM's 65.7%
  * This would be a MEANINGFUL architectural win, not just noise

The gap should come from:
  * Better long-range modeling (6 layers vs LSTM's single hidden state)
  * Richer representations (384-dim vs LSTM's 256-dim)
  * Specialized attention patterns (8 heads vs LSTM's uniform gate)

Run it:
    .venv/bin/python train_Transformer_improved.py
    .venv/bin/python train_Transformer_improved.py --d-model 512 --n-layers 8 --n-heads 16
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
# PHASE 1 QUICK WINS: Focal Loss, Label Smoothing
# --------------------------------------------------------------------------------------
def focal_loss(logits, targets, gamma=2.0, reduction='mean'):
    """
    Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Down-weights easy examples to focus training on hard decisions.
    gamma=0 -> standard cross-entropy
    gamma=2 -> strongly down-weight easy examples (recommended)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # p_t = softmax probability of correct class
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * ce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def label_smoothed_cross_entropy(logits, target, mask, smoothing=0.1):
    """Cross-entropy with label smoothing over valid actions only."""
    log_probs = F.log_softmax(logits, dim=1)

    # One-hot target (NLL loss)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

    # Smooth uniform distribution over valid options
    num_valid = mask.sum(1).float().clamp_min(1)
    smooth_dist = mask.float() / num_valid.unsqueeze(1)
    smooth_loss = -(log_probs * smooth_dist).sum(1)

    # Interpolate: (1-eps)*nll + eps*smooth
    loss = (1 - smoothing) * nll + smoothing * smooth_loss
    return loss.mean()


# --------------------------------------------------------------------------------------
# Dataset and batching: IDENTICAL to train_Transformer.py
# --------------------------------------------------------------------------------------
class GameDataset(Dataset):
    def __init__(self, state_n, ranges, offsets, dense_n, opt_card, opt_attack, y, games):
        self.state_n = state_n
        self.ranges = ranges
        self.offsets = offsets
        self.dense_n = dense_n
        self.opt_card = opt_card
        self.opt_attack = opt_attack
        self.y = y
        self.games = games

    def __len__(self):
        return len(self.games)

    def length_of(self, k):
        g = int(self.games[k]); a, b = self.ranges[g]
        return int(b - a)

    def __getitem__(self, k):
        g = int(self.games[k])
        a, b = int(self.ranges[g][0]), int(self.ranges[g][1])
        rows = range(a, b)
        state_seq = self.state_n[a:b]
        decs = []
        for i in rows:
            o0, o1 = int(self.offsets[i]), int(self.offsets[i + 1])
            decs.append((self.dense_n[o0:o1], self.opt_card[o0:o1],
                         self.opt_attack[o0:o1], int(self.y[i])))
        return state_seq, decs


def collate_games(batch):
    B = len(batch)
    lengths = [item[0].size(0) for item in batch]
    Tmax = max(lengths)
    Fs = batch[0][0].size(1)
    state_pad = torch.zeros(B, Tmax, Fs)
    seqmask = torch.zeros(B, Tmax, dtype=torch.bool)

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
    def __init__(self, dataset, batch_size, attn_budget=6_000_000, shuffle=True):
        """Increased attn_budget from 4M to 6M: with better batching we can afford it."""
        self.ds = dataset
        self.shuffle = shuffle
        order = sorted(range(len(dataset)), key=dataset.length_of)
        self.batches = []
        cur: list[int] = []
        for k in order:
            t = dataset.length_of(k)
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
# Positional encoding: same as original, but also support RoPE (future enhancement)
# --------------------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, T, device, dtype):
        calc = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        pos = torch.arange(T, device=device, dtype=calc)[:, None]
        i = torch.arange(0, self.d_model, 2, device=device, dtype=calc)
        freq = torch.exp(-math.log(10000.0) * i / self.d_model)
        pe = torch.zeros(T, self.d_model, device=device, dtype=calc)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)[:, : pe[:, 1::2].size(1)]
        return pe.to(dtype)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.table = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, T, device, dtype):
        if T > self.max_len:
            raise ValueError(
                f"Game has {T} decisions but --pos-max-len is {self.max_len}. "
                f"Raise it or use sinusoidal encoding.")
        return self.table(torch.arange(T, device=device)).to(dtype)


# --------------------------------------------------------------------------------------
# Improved multi-head attention with optional attention weight saving
# --------------------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
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
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x, attn_bias, return_attention=False):
        B, T, _ = x.shape
        q = self._heads(self.W_q(x), B, T)
        k = self._heads(self.W_k(x), B, T)
        v = self._heads(self.W_v(x), B, T)

        e = (q @ k.transpose(-2, -1)) * self.scale
        e = e + attn_bias
        alpha = self.attn_drop(e.softmax(dim=-1))
        a = alpha @ v
        a = a.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)

        out = self.resid_drop(self.W_o(a))
        if return_attention:
            return out, alpha.detach()
        return out


# --------------------------------------------------------------------------------------
# Improved Transformer block with PRE-NORM and optional attention saving
# --------------------------------------------------------------------------------------
class ImprovedTransformerBlock(nn.Module):
    """Pre-norm transformer block (the modern standard for deep models).

    Pre-norm (GPT-2/3, modern BERT):   x <- x + sublayer(LN(x))
    Post-norm (original Transformer):  x <- LN(x + sublayer(x))

    Pre-norm is more stable for deep stacks because gradients flow more directly
    through the residual connections. The original 2-layer model used post-norm to
    match lecture slides, but at 6 layers pre-norm is essential.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, use_gelu=True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # PHASE 1: Use GELU for smoother gradients (modern standard)
        activation = nn.GELU() if use_gelu else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias, return_attention=False):
        # Pre-norm: normalize BEFORE each sublayer
        if return_attention:
            attn_out, alpha = self.attn(self.norm1(x), attn_bias, return_attention=True)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x, alpha
        else:
            x = x + self.attn(self.norm1(x), attn_bias)
            x = x + self.ff(self.norm2(x))
            return x


def causal_padding_bias(seqmask, dtype):
    B, T = seqmask.shape
    device = seqmask.device
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    blocked = future[None, None] | (~seqmask)[:, None, None, :]
    return torch.zeros(B, 1, T, T, dtype=dtype, device=device).masked_fill_(
        blocked, torch.finfo(dtype).min)


# --------------------------------------------------------------------------------------
# PHASE 3: Option Cross-Attention - let options compare to each other
# --------------------------------------------------------------------------------------
class OptionCrossAttention(nn.Module):
    """Cross-attention between options for comparative scoring.

    THEORY:
    -------
    Standard scoring evaluates each option independently:
        score(option_i) = MLP(state || option_i)

    But many decisions involve comparing similar options:
        - "Retreat to Bench1" vs "Retreat to Bench2"
        - "Attack with move A" vs "Attack with move B"

    Cross-attention lets options attend to each other:
        option_i' = option_i + Attention(Q=option_i, K=all_options, V=all_options)

    This enables relative comparison: "option A is better BECAUSE of how it
    compares to option B" rather than scoring each in isolation.

    IMPLEMENTATION:
    ---------------
    Input: (B, M, D) option features, (B, M) validity mask
    Output: (B, M, D) attention-refined option features

    Uses standard multi-head attention with:
    - Query/Key/Value all from option features
    - Mask to ignore padding options
    - Residual connection + layer norm
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, option_features, option_mask):
        """
        Args:
            option_features: (B, M, D) - M options per batch item
            option_mask: (B, M) - True for valid options, False for padding

        Returns:
            (B, M, D) - refined option features
        """
        # key_padding_mask expects True for IGNORED positions (opposite of our mask)
        attn_out, _ = self.attn(
            option_features, option_features, option_features,
            key_padding_mask=~option_mask
        )

        # Residual + norm
        return self.norm(option_features + self.dropout(attn_out))


# --------------------------------------------------------------------------------------
# Improved Transformer model with deeper architecture
# --------------------------------------------------------------------------------------
class ImprovedTransformerCondLogit(nn.Module):
    def __init__(self, state_dim, opt_dim, card_vocab, attack_vocab,
                 d_model=384, n_heads=8, n_layers=6, d_ff=None, dropout=0.2,
                 head_hidden=(256,), card_emb=16, attack_emb=8,
                 learned_pos=False, pos_max_len=2048, use_gelu=True, input_noise=0.0,
                 use_option_cross_attn=False, option_cross_attn_heads=4):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # modern standard: 4x expansion

        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos = (LearnedPositionalEncoding(d_model, pos_max_len) if learned_pos
                    else SinusoidalPositionalEncoding(d_model))
        self.input_drop = nn.Dropout(dropout)

        # PHASE 1: Input noise for regularization
        self.input_noise = input_noise

        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(d_model, n_heads, d_ff, dropout, use_gelu=use_gelu)
            for _ in range(n_layers)
        ])

        # Pre-norm: final layer norm before output head
        self.final_norm = nn.LayerNorm(d_model)

        # Scoring head: identical to original
        self.card_embed = nn.Embedding(card_vocab + 1, card_emb, padding_idx=0)
        self.attack_embed = nn.Embedding(attack_vocab + 1, attack_emb, padding_idx=0)
        in_dim = d_model + opt_dim + card_emb + attack_emb

        # PHASE 3: Option cross-attention (optional)
        self.use_option_cross_attn = use_option_cross_attn
        if use_option_cross_attn:
            self.option_cross_attn = OptionCrossAttention(
                in_dim, n_heads=option_cross_attn_heads, dropout=dropout)

        layers: list[nn.Module] = []
        prev = in_dim
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def encode(self, state_pad, seqmask, return_attention=False):
        B, T, _ = state_pad.shape

        # PHASE 1: Input noise regularization (only during training)
        if self.training and self.input_noise > 0:
            state_pad = state_pad + torch.randn_like(state_pad) * self.input_noise

        x = self.input_proj(state_pad)
        x = x + self.pos(T, x.device, x.dtype)[None]
        x = self.input_drop(x)
        bias = causal_padding_bias(seqmask, x.dtype)

        attentions = [] if return_attention else None
        for block in self.blocks:
            if return_attention:
                x, alpha = block(x, bias, return_attention=True)
                attentions.append(alpha)
            else:
                x = block(x, bias)

        x = self.final_norm(x)
        if return_attention:
            return x, attentions
        return x

    def forward(self, state_pad, seqmask, dense, card, attack, optmask):
        # PHASE 1: Input noise for option features (only during training)
        if self.training and self.input_noise > 0:
            dense = dense + torch.randn_like(dense) * self.input_noise

        H = self.encode(state_pad, seqmask)
        h = H[seqmask]
        D, M, _ = dense.shape
        h_exp = h[:, None, :].expand(D, M, h.size(1))
        feat = torch.cat([h_exp, dense,
                          self.card_embed(card), self.attack_embed(attack)], dim=-1)

        # PHASE 3: Option cross-attention - let options compare to each other
        if self.use_option_cross_attn:
            feat = self.option_cross_attn(feat, optmask)

        scores = self.head(feat).squeeze(-1)
        return scores.masked_fill(~optmask, float("-inf"))


# --------------------------------------------------------------------------------------
# Cosine learning rate schedule with warmup
# --------------------------------------------------------------------------------------
class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to min_lr.

    This is the standard schedule for modern transformers (GPT, BERT, etc.).
    Better than flat LR because:
      * Warmup prevents early instability (random init + large LR = explosions)
      * Cosine decay helps convergence (smaller steps near the end find better minima)
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_mult = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_mult = self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_mult


# --------------------------------------------------------------------------------------
# Split and evaluation: same as original
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
    ap = argparse.ArgumentParser(
        description="Train the IMPROVED Transformer (deeper, wider, better schedule).")
    ap.add_argument("--data", default="dataset.pt")
    ap.add_argument("--out", default="Transformer_improved.pt")

    # Architecture: deeper, wider defaults
    ap.add_argument("--d-model", type=int, default=384,
                    help="model dimension (original: 192, improved: 384+)")
    ap.add_argument("--n-heads", type=int, default=8,
                    help="attention heads (original: 4, improved: 8+)")
    ap.add_argument("--n-layers", type=int, default=6,
                    help="transformer layers (original: 2, improved: 6+)")
    ap.add_argument("--d-ff", type=int, default=None,
                    help="feedforward dim (default: 4*d_model)")
    ap.add_argument("--head-hidden", type=int, nargs="+", default=[256])
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="dropout rate (original: 0.1, improved: 0.2)")

    # Positional encoding
    ap.add_argument("--learned-pos", action="store_true")
    ap.add_argument("--pos-max-len", type=int, default=2048)

    # Training: better schedule, longer training
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="weight decay (original: 1e-4, improved: 0.01)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="peak learning rate (original: 3e-4, improved: 1e-4)")
    ap.add_argument("--warmup-steps", type=int, default=1000,
                    help="warmup steps (original: 200, improved: 1000)")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="minimum LR for cosine decay")
    ap.add_argument("--epochs", type=int, default=40,
                    help="max epochs (original: 25, improved: 40)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--attn-budget", type=int, default=6_000_000,
                    help="attention budget (original: 4M, improved: 6M)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8,
                    help="early stopping patience (original: 5, improved: 8)")
    ap.add_argument("--min-delta", type=float, default=5e-4,
                    help="minimum improvement to reset patience (tightened)")
    ap.add_argument("--cpu", action="store_true")

    # Analysis
    ap.add_argument("--save-attention", action="store_true",
                    help="save attention weights for first validation batch (analysis)")

    # PHASE 1 QUICK WINS: New arguments
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="Focal loss gamma (0=CE, 2=focus on hard examples)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Label smoothing (0.1 recommended)")
    ap.add_argument("--input-noise", type=float, default=0.0,
                    help="Input feature noise (0.01 recommended)")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="Gradient accumulation steps (4 recommended)")
    ap.add_argument("--no-gelu", action="store_true",
                    help="Use ReLU instead of GELU in feedforward")

    # PHASE 3: Option cross-attention
    ap.add_argument("--use-option-cross-attn", action="store_true",
                    help="Enable cross-attention between options for comparative scoring")
    ap.add_argument("--option-cross-attn-heads", type=int, default=4,
                    help="Number of heads in option cross-attention")

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

    change = (torch.nonzero(game_index[1:] != game_index[:-1]).flatten() + 1)
    starts = torch.cat([torch.tensor([0]), change])
    ends = torch.cat([change, torch.tensor([game_index.numel()])])
    ranges = torch.stack([starts, ends], dim=1)

    g_tr, g_va, g_te = split_games(n_games, args.seed)
    def rows_of(games):
        m = torch.zeros(n_games, dtype=torch.bool); m[games] = True
        return torch.nonzero(m[game_index], as_tuple=True)[0]
    tr_rows = rows_of(g_tr)
    log(f"  games tr/va/te = {len(g_tr)}/{len(g_va)}/{len(g_te)}   "
        f"decisions train={len(tr_rows)}")

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

    model = ImprovedTransformerCondLogit(
        state.size(1), opt_dense.size(1), card_vocab_size, attack_vocab_size,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, head_hidden=tuple(args.head_hidden),
        learned_pos=args.learned_pos, pos_max_len=args.pos_max_len,
        use_gelu=not args.no_gelu, input_noise=args.input_noise,
        use_option_cross_attn=args.use_option_cross_attn,
        option_cross_attn_heads=args.option_cross_attn_heads).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"\n=== MODEL ARCHITECTURE ===")
    log(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    log(f"  d_ff={args.d_ff or 4*args.d_model}  d_head={args.d_model//args.n_heads}")
    log(f"  dropout={args.dropout}  pos={'learned' if args.learned_pos else 'sin/cos'}")
    log(f"  head_hidden={args.head_hidden}  activation={'ReLU' if args.no_gelu else 'GELU'}")
    log(f"  option_cross_attn={'ON' if args.use_option_cross_attn else 'OFF'}")
    log(f"  total params: {n_params:,}")
    log(f"\n=== TRAINING CONFIG ===")
    log(f"  lr={args.lr}  warmup={args.warmup_steps}  weight_decay={args.weight_decay}")
    log(f"  epochs={args.epochs}  patience={args.patience}  batch_size={args.batch_size}")
    log(f"  clip={args.clip}  seed={args.seed}")
    log(f"  Quick wins: focal_gamma={args.focal_gamma}  label_smooth={args.label_smoothing}  "
        f"input_noise={args.input_noise}  accum_steps={args.accum_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Estimate total steps for scheduler
    total_steps = len(train_b) * args.epochs
    scheduler = CosineWarmupScheduler(opt, args.warmup_steps, total_steps, args.min_lr)

    step = 0
    best_val, best_state, since = -1.0, None, 0

    log("\n=== TRAINING ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = seen = 0
        accum_counter = 0
        opt.zero_grad()  # Zero gradients once at the start of epoch

        for batch in train_b:
            sp, sm, ln, dense, card, attack, om, yb = collate_games(batch)
            sp, sm, dense = sp.to(device), sm.to(device), dense.to(device)
            card, attack, om, yb = card.to(device), attack.to(device), om.to(device), yb.to(device)

            scores = model(sp, sm, dense, card, attack, om)

            # PHASE 1: Use focal loss and/or label smoothing
            if args.label_smoothing > 0:
                loss = label_smoothed_cross_entropy(scores, yb, om, args.label_smoothing)
            elif args.focal_gamma > 0:
                loss = focal_loss(scores, yb, gamma=args.focal_gamma)
            else:
                loss = F.cross_entropy(scores, yb)

            # Scale loss for gradient accumulation
            loss = loss / args.accum_steps
            loss.backward()

            accum_counter += 1
            if accum_counter == args.accum_steps:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                scheduler.step()
                step += 1
                opt.zero_grad()
                accum_counter = 0

            run_loss += loss.item() * args.accum_steps * yb.numel()
            seen += yb.numel()

        # Handle remaining gradients at end of epoch
        if accum_counter > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            scheduler.step()
            step += 1
            opt.zero_grad()

        v1, v3, vloss = evaluate(model, val_b, device)
        improved = v1 > best_val + args.min_delta
        if improved:
            best_val = v1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1

        current_lr = opt.param_groups[0]["lr"]
        log(f"epoch {epoch:>3}  train_loss={run_loss/seen:.4f}  val_loss={vloss:.4f}  "
            f"val_top1={v1:.4f}  val_top3={v3:.4f}  lr={current_lr:.2e}"
            f"{'  *' if improved else ''}  ({time.time()-t0:.1f}s)")

        if since >= args.patience:
            log(f"Early stop: val_top1 gained < {args.min_delta} for {args.patience} "
                f"straight epochs (best {best_val:.4f}).")
            break

    model.load_state_dict(best_state)
    t1, t3, tloss = evaluate(model, test_b, device)

    log("\n=== RESULTS ===")
    log(f"  best val top1  : {best_val:.4f}")
    log(f"  TEST top1      : {t1:.4f}   top3: {t3:.4f}   (loss {tloss:.4f})")
    log(f"  vs baselines   : always-0={b0:.3f}  random-legal={brand:.3f}")
    log(f"  improvement    : +{(t1-b0)*100:.1f} pts over always-0")
    log(f"\n=== COMPARISON TO BASELINES ===")
    log(f"  LSTM (reported)      : 65.7% test accuracy")
    log(f"  Transformer (orig)   : 66.6% test accuracy")
    log(f"  Transformer (improved): {t1*100:.1f}% test accuracy")
    log(f"  Gain over LSTM       : +{(t1-0.657)*100:.1f} pts")
    log(f"  Gain over shallow    : +{(t1-0.666)*100:.1f} pts")

    if save:
        torch.save({
            "state_dict": best_state, "config": vars(args),
            "state_mean": s_mean, "state_std": s_std, "opt_mean": o_mean, "opt_std": o_std,
            "card_vocab_size": card_vocab_size, "attack_vocab_size": attack_vocab_size,
            "test_acc": t1, "test_top3": t3, "val_acc": best_val,
        }, args.out)
        log(f"\n  saved -> {args.out}")

    return {"seed": args.seed, "val_top1": best_val, "test_top1": t1, "test_top3": t3,
            "baseline_always0": b0, "baseline_random": brand}


def main():
    train_and_eval(build_parser().parse_args(), save=True, verbose=True)


if __name__ == "__main__":
    main()
