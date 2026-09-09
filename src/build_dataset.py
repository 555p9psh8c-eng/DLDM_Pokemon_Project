"""build_dataset.py — dataset for the CONDITIONAL-LOGIT model.

The positional dataset (build_dataset.py -> dataset.pt) stores only the BOARD state and
the chosen INDEX. That's enough for a model that predicts "which slot in the menu", but
not for one that reasons about WHAT each option does. This script additionally records,
for every legal option of every decision, a small feature vector (type / target / card /
attack) so a conditional-logit model can score each candidate action on its merits.

Because the number of options varies per decision, the options are stored in a flat
"ragged" (CSR) layout: one big table of option rows, plus an offsets array that says
which rows belong to which decision. This avoids padding on disk.

Run it (from the folder that contains episodes/):

    ../.venv/bin/python build_dataset.py                 # all episodes
    ../.venv/bin/python build_dataset.py --limit 200     # quick test

What ends up in dataset.pt (a dict you load with torch.load):
    state         : float32 (N, F)     board state per decision (same as dataset.pt's X)
    offsets       : int64   (N+1,)      CSR offsets; decision i's options = [off[i]:off[i+1]]
    opt_dense     : float32 (T, Fo)     dense per-option features (T = total options)
    opt_card      : int64   (T,)        hand-card vocab index (+1); 0 = none/OOV
    opt_attack    : int64   (T,)        attack vocab index (+1); 0 = none/OOV
    y             : int64   (N,)        chosen option position within its decision
    num_options   : int64   (N,)        == off[i+1]-off[i]
    turn          : int64   (N,)
    game_index    : int64   (N,)
    game_ids      : list[str]
    feature_names / option_feature_names : list[str]
    card_vocab    : {card_id: col}      (reused from dataset.pt)
    attack_vocab  : {attackId: col}
    meta          : dict
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from feature_extractor import (
    FeatureExtractor, build_attack_vocab, build_card_vocab, iter_main_decisions,
    option_features, option_feature_names, NUM_OPTION_FEATURES,
    load_card_vocab, save_card_vocab,
)


def get_card_vocab(path_json: str, episodes_glob: str) -> dict:
    """Load the hand-card vocabulary from its JSON dump, or build it fresh from the
    episodes (and save it) if the JSON isn't there. Self-contained — no dependency on
    any other dataset file."""
    if os.path.exists(path_json):
        print(f"Card vocab: loading {path_json}")
        return load_card_vocab(path_json)
    print("Card vocab: JSON not found — building from episodes (one scan)...")
    vocab = build_card_vocab(episodes_glob, min_count=20, max_cards=300)
    save_card_vocab(vocab, path_json)
    print(f"  built and saved {len(vocab)} cards -> {path_json}")
    return vocab


def main():
    ap = argparse.ArgumentParser(description="Build dataset.pt (per-option features).")
    ap.add_argument("--episodes", default="episodes/*.json")
    ap.add_argument("--out", default="dataset.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--card-vocab-json", default="dataset_card_vocab.json")
    ap.add_argument("--attack-min-count", type=int, default=20,
                    help="min times an attackId must appear to get an embedding slot")
    ap.add_argument("--all-players", action="store_true",
                    help="learn from BOTH players (default: winner only, matches dataset.pt)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.episodes))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No episodes matched {args.episodes!r}.")
    print(f"Found {len(files)} episodes.")
    winners_only = not args.all_players

    # --- vocabularies ---
    card_vocab = get_card_vocab(args.card_vocab_json, args.episodes)
    print(f"Card vocab: {len(card_vocab)} cards.")

    print("Building attack vocabulary (one scan)...")
    t0 = time.time()
    glob_for_vocab = args.episodes
    if args.limit:                     # only scan the same subset when limiting
        # build_attack_vocab takes a glob; emulate the limit by a temp scan over `files`.
        import collections
        counts = collections.Counter()
        for path in files:
            for dec in iter_main_decisions(path, winners_only=winners_only):
                for o in dec["options"]:
                    if o.get("type") == 13 and o.get("attackId") is not None:
                        counts[o["attackId"]] += 1
        attack_vocab = {a: i for i, a in enumerate(
            sorted(a for a, c in counts.items() if c >= args.attack_min_count))}
    else:
        attack_vocab = build_attack_vocab(glob_for_vocab, min_count=args.attack_min_count,
                                          winners_only=winners_only)
    print(f"  attack vocab: {len(attack_vocab)} attacks  ({time.time()-t0:.1f}s)")

    fe = FeatureExtractor(card_vocab=card_vocab)
    print(f"State feature dim: {fe.dim}   option feature dim: {NUM_OPTION_FEATURES}")

    # --- accumulate ---
    state_rows: list = []
    y_rows: list = []
    turn_rows: list = []
    gidx_rows: list = []
    offsets: list = [0]
    opt_dense: list = []
    opt_card: list = []
    opt_attack: list = []
    game_ids: list = []
    game_id_to_idx: dict = {}

    t0 = time.time()
    skipped = 0
    for fi, path in enumerate(files):
        try:
            decisions = list(iter_main_decisions(path, winners_only=winners_only))
        except Exception:
            skipped += 1
            continue
        for dec in decisions:
            cur = (dec["observation"].get("current") or {})
            players = cur.get("players") or []
            mi = int(cur.get("yourIndex", 0))
            me = players[mi] if mi < len(players) else {}
            opp = players[1 - mi] if (1 - mi) < len(players) else {}

            state_rows.append(fe.extract(dec["observation"]))
            y_rows.append(dec["chosen"])
            turn_rows.append(dec["turn"])
            no = dec["num_options"]

            gid = dec["game_id"]
            gi = game_id_to_idx.get(gid)
            if gi is None:
                gi = len(game_ids)
                game_id_to_idx[gid] = gi
                game_ids.append(gid)
            gidx_rows.append(gi)

            for pos, o in enumerate(dec["options"]):
                dense, ci, ai = option_features(o, me, opp, card_vocab, attack_vocab, pos, no)
                opt_dense.append(dense)
                opt_card.append(ci)
                opt_attack.append(ai)
            offsets.append(len(opt_dense))
        if (fi + 1) % 250 == 0 or fi + 1 == len(files):
            rate = (fi + 1) / (time.time() - t0)
            print(f"  {fi+1:>5}/{len(files)} episodes | {len(state_rows):>7} decisions | "
                  f"{len(opt_dense):>8} options | {rate:.0f} eps/s"
                  + (f" | {skipped} skipped" if skipped else ""))

    if not state_rows:
        raise SystemExit("No decisions extracted — check the episode format.")

    # --- to tensors ---
    print("Stacking into tensors...")
    dataset = {
        "state": torch.from_numpy(np.asarray(state_rows, dtype=np.float32)),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "opt_dense": torch.from_numpy(np.asarray(opt_dense, dtype=np.float32)),
        "opt_card": torch.tensor(opt_card, dtype=torch.long),
        "opt_attack": torch.tensor(opt_attack, dtype=torch.long),
        "y": torch.tensor(y_rows, dtype=torch.long),
        "num_options": torch.tensor([offsets[i + 1] - offsets[i] for i in range(len(y_rows))],
                                    dtype=torch.long),
        "turn": torch.tensor(turn_rows, dtype=torch.long),
        "game_index": torch.tensor(gidx_rows, dtype=torch.long),
        "game_ids": game_ids,
        "feature_names": fe.feature_names(),
        "option_feature_names": option_feature_names(),
        "card_vocab": card_vocab,
        "attack_vocab": attack_vocab,
        "meta": {
            "num_decisions": len(y_rows),
            "num_options_total": len(opt_dense),
            "state_dim": fe.dim,
            "option_dim": NUM_OPTION_FEATURES,
            "card_vocab_size": len(card_vocab),
            "attack_vocab_size": len(attack_vocab),
            "num_games": len(game_ids),
            "winners_only": winners_only,
            "episodes_used": len(files),
        },
    }
    torch.save(dataset, args.out)

    size_mb = os.path.getsize(args.out) / 1e6
    y = dataset["y"]
    print("\n=== dataset.pt written ===")
    print(f"  file           : {args.out}  ({size_mb:.1f} MB)")
    print(f"  state          : {tuple(dataset['state'].shape)}")
    print(f"  opt_dense      : {tuple(dataset['opt_dense'].shape)}")
    print(f"  decisions      : {len(y_rows)}   options total: {len(opt_dense)}")
    print(f"  games          : {len(game_ids)}   attacks: {len(attack_vocab)}")
    print(f"  always-index-0 : {(y == 0).float().mean().item()*100:.1f}%")
    print(f"  built in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
