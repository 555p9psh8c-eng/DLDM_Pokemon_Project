"""feature_extractor.py — turn a raw Pokémon-TCG observation into a flat numeric vector.

This is Step 1 of the project pipeline (see PROJECT_GUIDE.md). Every model we train
(MLP, LSTM, Transformer) consumes the SAME feature vector produced here, so this file
is the single definition of "what the network sees" at each decision point.

--------------------------------------------------------------------------------------
WHAT A DECISION LOOKS LIKE IN THE RAW DATA
--------------------------------------------------------------------------------------
Each episode JSON has `steps[t][player]['observation']`. The observation has:
    observation['current']        -> the whole board state (dict, described below)
    observation['select']         -> the decision being asked (context + list of options)
    observation['current']['yourIndex'] -> which player index (0/1) is "me"

`current` layout (the part we featurize):
    current['turn']               int, turn counter
    current['firstPlayer']        0/1, who went first this game
    current['yourIndex']          0/1, which player object below is "me"
    current['energyAttached']     bool, did I already attach an energy this turn
    current['supporterPlayed']    bool, did I already play a supporter this turn
    current['stadiumPlayed']      bool
    current['stadium']            list, the stadium card in play (empty = none)
    current['players'][i]         one dict per player:
        active   : [mon]          my/opp active Pokémon (0 or 1 entries)
        bench    : [mon, ...]     up to 5 benched Pokémon
        benchMax : int            usually 5
        hand     : [card,...] or None   (None when it's the opponent — hidden)
        handCount: int            number of cards in hand (visible for BOTH players)
        deckCount: int            cards left in deck
        prize    : [null,...]     6-slot list; count of non-null = prizes REMAINING
        poisoned/burned/asleep/paralyzed/confused : bool status flags

    a `mon` (Pokémon) dict:
        id        : int           card id (identity of the Pokémon)
        hp        : int           current HP
        maxHp     : int           max HP
        energies  : [int,...]     attached energy, each int is an EnergyType
        tools     : [...]         attached tool cards (empty = none)

The LABEL for training is the index of the option the expert chose (handled by
`iter_main_decisions`, not by the feature vector itself).

--------------------------------------------------------------------------------------
DESIGN CHOICES
--------------------------------------------------------------------------------------
* Fixed-length output. The number of legal options changes every turn, but the BOARD
  STATE has a fixed shape, so this vector always has the same length `dim`. (The models
  handle the variable option count in their output layer, not here.)
* HP is stored as a RATIO (hp/maxHp) so a 320-HP tank and a 70-HP attacker are on the
  same 0..1 scale — networks train better on normalized inputs.
* Counts (deck, hand, energies, prizes, turn) are divided by sensible maxima so they
  also land roughly in 0..1.
* Card identity: a Pokémon's `id` is encoded as an energy-type histogram + numbers, and
  the HAND is encoded as an optional multi-hot "bag of cards" over a vocabulary you can
  build with `build_card_vocab`. Without a vocab you still get a fully usable vector;
  the hand is then summarized by counts only.

Everything is numpy-only. No engine / cg-lib dependency, so it runs directly on the
downloaded episode JSONs.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Iterator, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# Constants describing the game (from CLAUDE.md "Engine API essentials")
# --------------------------------------------------------------------------------------

# EnergyType ids we bucket into a histogram. 0..7 cover the standard types; anything
# outside (special energies like Ignition=17, Mist=11, Telepath=19) folds into "other".
ENERGY_TYPES = {
    0: "colorless",
    1: "grass",
    2: "fire",
    3: "water",
    4: "lightning",
    5: "psychic",
    6: "fighting",
    7: "darkness",
}
NUM_ENERGY_BUCKETS = len(ENERGY_TYPES) + 1  # +1 "other" bucket for special energies

STATUS_FLAGS = ("poisoned", "burned", "asleep", "paralyzed", "confused")

# SelectContext values (only MAIN=0 are real strategic decisions we train on).
CONTEXT_MAIN = 0

# Normalization constants — rough game maxima so features land in ~[0, 1].
MAX_BENCH = 5
MAX_PRIZE = 6
MAX_ENERGY_ON_MON = 6.0     # attaching more than ~6 energy is very rare
MAX_HAND = 12.0             # hands above ~12 cards are rare
MAX_DECK = 60.0             # a full deck is 60 cards
MAX_TURN = 40.0             # games rarely pass ~40 turns


# --------------------------------------------------------------------------------------
# Card vocabulary (optional) — lets the hand become a multi-hot "bag of cards"
# --------------------------------------------------------------------------------------

def build_card_vocab(episode_glob: str, min_count: int = 20, max_cards: int = 400) -> dict:
    """Scan episodes and return a {card_id: column_index} mapping for the hand bag.

    Only cards seen at least `min_count` times (across all hands) are kept, capped at
    `max_cards` most-frequent cards. Rare cards collapse into an implicit "unknown"
    (they simply don't get a column), which keeps the vector compact and stable.

    Pass the returned dict to FeatureExtractor(card_vocab=...).
    """
    counts: Counter = Counter()
    for path in glob.glob(episode_glob):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        for step in data.get("steps", []):
            for ent in step:
                cur = (ent.get("observation") or {}).get("current")
                if not cur:
                    continue
                for p in cur.get("players", []) or []:
                    if not p:
                        continue
                    hand = p.get("hand")
                    if not hand:      # None for opponent, or empty
                        continue
                    for card in hand:
                        if card and "id" in card:
                            counts[card["id"]] += 1
    common = [cid for cid, c in counts.most_common(max_cards) if c >= min_count]
    return {cid: i for i, cid in enumerate(sorted(common))}


def save_card_vocab(vocab: dict, path: str) -> None:
    json.dump({str(k): v for k, v in vocab.items()}, open(path, "w"), indent=0)


def load_card_vocab(path: str) -> dict:
    return {int(k): v for k, v in json.load(open(path)).items()}


# --------------------------------------------------------------------------------------
# The feature extractor
# --------------------------------------------------------------------------------------

class FeatureExtractor:
    """Converts one observation dict into a fixed-length float32 numpy vector.

    Usage:
        fe = FeatureExtractor()                     # no card vocab
        # or: fe = FeatureExtractor(build_card_vocab('episodes/*.json'))
        x = fe.extract(observation_dict)            # -> np.ndarray shape (fe.dim,)
        names = fe.feature_names()                  # human-readable label per index
        assert len(names) == fe.dim == x.shape[0]
    """

    def __init__(self, card_vocab: Optional[dict] = None):
        self.card_vocab = card_vocab or {}
        self.vocab_size = len(self.card_vocab)
        self._names = self._build_names()
        self.dim = len(self._names)

    # ---- public API -------------------------------------------------------------

    def feature_names(self) -> list:
        """Human-readable name for every index in the output vector."""
        return list(self._names)

    def extract(self, observation: dict) -> np.ndarray:
        """Build the feature vector for a single observation dict.

        `observation` may be the full observation (with 'current') or the 'current'
        board dict directly — both are accepted.
        """
        cur = observation.get("current", observation)
        feats: list = []

        me_idx = int(cur.get("yourIndex", 0))
        players = cur.get("players", []) or []
        me = players[me_idx] if me_idx < len(players) else {}
        opp = players[1 - me_idx] if (1 - me_idx) < len(players) else {}

        # --- global / turn context ---
        first_player = cur.get("firstPlayer", 0)
        feats.append(min(cur.get("turn", 0) / MAX_TURN, 1.0))          # turn (normalized)
        feats.append(1.0 if first_player == me_idx else 0.0)          # do I have first-player edge
        feats.append(1.0 if cur.get("energyAttached") else 0.0)       # already attached energy
        feats.append(1.0 if cur.get("supporterPlayed") else 0.0)      # already played supporter
        feats.append(1.0 if cur.get("stadiumPlayed") else 0.0)        # already played stadium
        feats.append(1.0 if (cur.get("stadium") or []) else 0.0)      # a stadium is in play

        # --- my side (full info: hand cards visible) ---
        feats.extend(self._player_block(me, include_hand_bag=True))
        # --- opponent side (hand hidden: only counts) ---
        feats.extend(self._player_block(opp, include_hand_bag=False))

        arr = np.asarray(feats, dtype=np.float32)
        # Safety: guarantee fixed length even on malformed observations.
        if arr.shape[0] != self.dim:
            fixed = np.zeros(self.dim, dtype=np.float32)
            fixed[: min(self.dim, arr.shape[0])] = arr[: self.dim]
            return fixed
        return arr

    # ---- per-player block -------------------------------------------------------

    def _player_block(self, player: dict, include_hand_bag: bool) -> list:
        """Features for one player. `include_hand_bag` adds the multi-hot hand vector
        (only meaningful for MY hand; the opponent's hand is hidden)."""
        out: list = []

        # Active Pokémon (0 or 1 present).
        active_list = player.get("active") or []
        active = active_list[0] if active_list else None
        out.extend(self._mon_block(active))

        # Bench: exactly MAX_BENCH slots, padded with empties for a fixed shape.
        bench = player.get("bench") or []
        for i in range(MAX_BENCH):
            out.extend(self._mon_block(bench[i] if i < len(bench) else None, compact=True))

        # Prizes remaining (0..6): a non-null slot means a prize is still uncollected.
        prize = player.get("prize") or []
        prizes_remaining = sum(1 for x in prize if x is not None)
        out.append(prizes_remaining / MAX_PRIZE)

        # Resource counts.
        out.append(min(player.get("deckCount", 0) / MAX_DECK, 1.0))
        out.append(min(player.get("handCount", 0) / MAX_HAND, 1.0))
        out.append(min(len(player.get("bench") or []) / MAX_BENCH, 1.0))

        # Status conditions on this player.
        for flag in STATUS_FLAGS:
            out.append(1.0 if player.get(flag) else 0.0)

        # Multi-hot bag of MY hand cards (skipped for opponent / when no vocab).
        if include_hand_bag and self.vocab_size:
            bag = np.zeros(self.vocab_size, dtype=np.float32)
            for card in (player.get("hand") or []):
                idx = self.card_vocab.get(card.get("id")) if card else None
                if idx is not None:
                    bag[idx] += 1.0
            out.extend(bag.tolist())

        return out

    # ---- per-Pokémon block ------------------------------------------------------

    def _mon_block(self, mon: Optional[dict], compact: bool = False) -> list:
        """Features for a single Pokémon slot.

        compact=True (bench slots) drops the per-type energy histogram to keep the
        vector small — bench mons mostly matter by "is it there / how healthy / how
        fueled". The active Pokémon gets the full histogram.
        """
        if mon is None:
            # Empty slot: present flag 0, everything else 0.
            if compact:
                return [0.0, 0.0, 0.0, 0.0]  # present, hp_ratio, energy_count, has_tool
            return [0.0, 0.0, 0.0, 0.0] + [0.0] * NUM_ENERGY_BUCKETS

        present = 1.0
        max_hp = mon.get("maxHp") or mon.get("hp") or 1
        hp_ratio = (mon.get("hp", 0) / max_hp) if max_hp else 0.0
        energies = mon.get("energies") or []
        energy_count = min(len(energies) / MAX_ENERGY_ON_MON, 1.0)
        has_tool = 1.0 if (mon.get("tools") or []) else 0.0

        block = [present, float(np.clip(hp_ratio, 0.0, 1.0)), energy_count, has_tool]
        if compact:
            return block

        # Energy-type histogram (fraction of attached energy of each type).
        hist = np.zeros(NUM_ENERGY_BUCKETS, dtype=np.float32)
        for e in energies:
            bucket = e if e in ENERGY_TYPES else (NUM_ENERGY_BUCKETS - 1)
            hist[bucket] += 1.0
        if energies:
            hist /= len(energies)
        return block + hist.tolist()

    # ---- name bookkeeping (must mirror the order built above) -------------------

    def _mon_names(self, prefix: str, compact: bool) -> list:
        base = [f"{prefix}.present", f"{prefix}.hp_ratio",
                f"{prefix}.energy_count", f"{prefix}.has_tool"]
        if compact:
            return base
        return base + [f"{prefix}.energy[{ENERGY_TYPES.get(i, 'other')}]"
                       for i in range(NUM_ENERGY_BUCKETS)]

    def _player_names(self, side: str, include_hand_bag: bool) -> list:
        names = self._mon_names(f"{side}.active", compact=False)
        for i in range(MAX_BENCH):
            names += self._mon_names(f"{side}.bench{i}", compact=True)
        names += [f"{side}.prizes_remaining", f"{side}.deck_count",
                  f"{side}.hand_count", f"{side}.bench_fill"]
        names += [f"{side}.status.{f}" for f in STATUS_FLAGS]
        if include_hand_bag and self.vocab_size:
            inv = {v: k for k, v in self.card_vocab.items()}
            names += [f"{side}.hand_has[card{inv[i]}]" for i in range(self.vocab_size)]
        return names

    def _build_names(self) -> list:
        names = ["turn", "i_go_first", "energy_attached", "supporter_played",
                 "stadium_played", "stadium_in_play"]
        names += self._player_names("me", include_hand_bag=True)
        names += self._player_names("opp", include_hand_bag=False)
        return names


# --------------------------------------------------------------------------------------
# Per-option (per-action) features — for the CONDITIONAL-LOGIT model
# --------------------------------------------------------------------------------------
#
# The positional MLP scores an option purely by its INDEX in the menu. The conditional
# logit instead scores WHAT EACH OPTION DOES. So for every legal option we build a small
# feature vector describing: its type (play/attach/attack/...), the card or attack it
# uses, and the state of the Pokémon it targets. The model concatenates this with the
# shared board vector and produces one score per option, then softmaxes over the legal
# set (exactly the structure of research/train_bc.py, generalized from linear to an MLP).
#
# Two high-cardinality ids (the hand card and the attack) are returned as integer indices
# for the model to EMBED, rather than one-hot, to keep the tensors compact.

# OptionType ids that appear in MAIN decisions, in a fixed slot order for the one-hot.
OPTION_TYPES = [7, 8, 9, 10, 12, 13, 14]   # PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END
OPTION_TYPE_NAMES = ["PLAY", "ATTACH", "EVOLVE", "ABILITY", "RETREAT", "ATTACK", "END", "OTHER"]
NUM_OPTION_TYPES = len(OPTION_TYPES) + 1    # +1 "other" slot for anything unexpected

# OptionType.END (used to detect the "end turn" choice when the action list is empty).
OPTION_END = 14


def option_feature_names() -> list:
    """Human-readable name per column of the dense per-option feature vector."""
    names = [f"opt.type[{n}]" for n in OPTION_TYPE_NAMES]
    names += ["opt.pos_norm", "opt.target_active", "opt.target_bench",
              "opt.target_hp_ratio", "opt.target_energy"]
    return names


NUM_OPTION_FEATURES = len(option_feature_names())   # 8 + 5 = 13


def _mon_hp_energy(mon: Optional[dict]) -> tuple:
    """(hp_ratio, energy_count_norm) for a Pokémon dict, or (0, 0) if absent."""
    if not mon:
        return 0.0, 0.0
    max_hp = mon.get("maxHp") or mon.get("hp") or 1
    hp_ratio = float(np.clip((mon.get("hp", 0) / max_hp) if max_hp else 0.0, 0.0, 1.0))
    energy = min(len(mon.get("energies") or []) / MAX_ENERGY_ON_MON, 1.0)
    return hp_ratio, energy


def option_features(option: dict, me: dict, opp: dict, card_vocab: dict,
                    attack_vocab: dict, pos: int, num_options: int) -> tuple:
    """Encode ONE legal option.

    Returns (dense, card_idx, attack_idx):
        dense       : list[float] of length NUM_OPTION_FEATURES
        card_idx    : int   0 = none/out-of-vocab, else vocab index + 1 (for embedding)
        attack_idx  : int   0 = none/out-of-vocab, else attack-vocab index + 1

    `me`/`opp` are the current player's / opponent's player dicts (from current.players).
    """
    t = option.get("type")
    slot = OPTION_TYPES.index(t) if t in OPTION_TYPES else (NUM_OPTION_TYPES - 1)
    onehot = [0.0] * NUM_OPTION_TYPES
    onehot[slot] = 1.0

    pos_norm = pos / max(num_options - 1, 1)
    ipa = option.get("inPlayArea")
    target_active = 1.0 if ipa == 4 else 0.0
    target_bench = 1.0 if ipa == 5 else 0.0

    # State of the Pokémon this option targets.
    if t == 13:                       # ATTACK -> the opponent's active is the target
        act = (opp.get("active") or [None])
        hp_ratio, energy = _mon_hp_energy(act[0] if act else None)
    elif ipa == 4:                    # my active
        act = (me.get("active") or [None])
        hp_ratio, energy = _mon_hp_energy(act[0] if act else None)
    elif ipa == 5:                    # my bench slot
        bench = me.get("bench") or []
        ii = option.get("inPlayIndex", 0)
        hp_ratio, energy = _mon_hp_energy(bench[ii] if 0 <= ii < len(bench) else None)
    else:
        hp_ratio, energy = 0.0, 0.0

    dense = onehot + [pos_norm, target_active, target_bench, hp_ratio, energy]

    # Card identity: PLAY/ATTACH/EVOLVE carry a hand index -> look up the card id.
    card_idx = 0
    if t in (7, 8, 9) and "index" in option:
        hand = me.get("hand") or []
        i = option["index"]
        if 0 <= i < len(hand) and hand[i]:
            v = card_vocab.get(hand[i].get("id"))
            if v is not None:
                card_idx = v + 1

    # Attack identity: ATTACK carries an attackId.
    attack_idx = 0
    if t == 13:
        v = attack_vocab.get(option.get("attackId"))
        if v is not None:
            attack_idx = v + 1

    return dense, card_idx, attack_idx


def build_attack_vocab(episode_glob: str, min_count: int = 20,
                       winners_only: bool = True) -> dict:
    """Scan episodes and return {attackId: column_index} for the ATTACK options."""
    counts: Counter = Counter()
    for path in glob.glob(episode_glob):
        for dec in iter_main_decisions(path, winners_only=winners_only):
            for o in dec["options"]:
                if o.get("type") == 13 and o.get("attackId") is not None:
                    counts[o["attackId"]] += 1
    common = [aid for aid, c in counts.most_common() if c >= min_count]
    return {aid: i for i, aid in enumerate(sorted(common))}


# --------------------------------------------------------------------------------------
# Decision iterator — pairs each MAIN board state with the expert's chosen option index
# --------------------------------------------------------------------------------------


def _chosen_index(action, options) -> Optional[int]:
    """Recover the option index the expert picked (mirrors research/train_bc.py).

    * A single explicit index -> that index.
    * An empty action -> the END option (they chose to end the turn).
    * Multi-select actions are skipped (return None).
    """
    if action and len(action) == 1:
        idx = action[0]
        return idx if 0 <= idx < len(options) else None
    if not action:
        for i, o in enumerate(options):
            if o.get("type") == OPTION_END:
                return i
        return None
    return None  # multi-select — skip


def iter_main_decisions(episode_path: str, winners_only: bool = True) -> Iterator[dict]:
    """Yield one dict per MAIN decision in an episode.

    Each yielded dict:
        {
          'observation': <observation dict>,   # feed to FeatureExtractor.extract
          'options':     <list[dict]>,          # raw legal options (feed to option_features)
          'chosen':      <int>,                 # label: index of the expert's option
          'num_options': <int>,                 # how many legal options there were
          'turn':        <int>,
          'player':      <int>,
          'game_id':     <str>,                 # episode id (for grouping / splitting)
        }

    winners_only=True keeps only decisions made by the game's winner (the same
    imitation-learning choice the repo's train_bc.py makes — learn from good play).
    """
    try:
        data = json.load(open(episode_path))
    except Exception:
        return
    rewards = data.get("rewards") or []
    if len(rewards) < 2:
        return
    r0, r1 = rewards[0], rewards[1]
    if r0 is None or r1 is None:
        # A game that errored / didn't finish cleanly has no clear winner.
        if winners_only:
            return
        winner = -1
    else:
        winner = 0 if r0 > r1 else (1 if r1 > r0 else -1)
    game_id = str(data.get("info", {}).get("EpisodeId", os.path.basename(episode_path)))

    players_to_keep = ({winner} if winners_only else {0, 1})
    if winners_only and winner < 0:
        return  # a draw: no clear expert to imitate

    for step in data.get("steps", []):
        for pi, ent in enumerate(step):
            if pi not in players_to_keep:
                continue
            od = ent.get("observation") or {}
            sel = od.get("select")
            if not sel or sel.get("context") != CONTEXT_MAIN:
                continue
            options = sel.get("option") or []
            if len(options) < 2:
                continue  # forced move — no real decision
            chosen = _chosen_index(ent.get("action") or [], options)
            if chosen is None:
                continue
            cur = od.get("current") or {}
            yield {
                "observation": od,
                "options": options,                  # raw legal options (for per-option feats)
                "chosen": chosen,
                "num_options": len(options),
                "turn": cur.get("turn", 0),
                "player": pi,
                "game_id": game_id,
            }


# --------------------------------------------------------------------------------------
# CLI: quick self-test / summary when run directly
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Feature extractor self-test / summary.")
    ap.add_argument("--episodes", default="episodes/*.json",
                    help="glob for episode JSONs (default: episodes/*.json)")
    ap.add_argument("--limit", type=int, default=25, help="episodes to sample")
    ap.add_argument("--vocab", action="store_true",
                    help="build a card vocab from the sampled episodes first")
    args = ap.parse_args()

    files = sorted(glob.glob(args.episodes))[: args.limit]
    if not files:
        raise SystemExit(f"No episodes matched {args.episodes!r} "
                         f"(run from the folder that contains 'episodes/').")

    vocab = None
    if args.vocab:
        # Build from the sampled files only (fast); use the full glob for the real run.
        tmp_glob = os.path.join(os.path.dirname(files[0]) or ".", "*.json")
        vocab = build_card_vocab(tmp_glob, min_count=5, max_cards=300)
        print(f"Built card vocab: {len(vocab)} cards")

    fe = FeatureExtractor(card_vocab=vocab)
    print(f"Feature vector dimension: {fe.dim}")

    n = 0
    example = None
    for path in files:
        for dec in iter_main_decisions(path):
            x = fe.extract(dec["observation"])
            if example is None:
                example = (dec, x)
            n += 1
    print(f"Extracted {n} MAIN decisions from {len(files)} episodes.")
    if example is not None:
        dec, x = example
        print(f"\nExample decision (game {dec['game_id']}, turn {dec['turn']}): "
              f"{dec['num_options']} options, expert chose index {dec['chosen']}")
        print(f"Feature vector shape: {x.shape}, "
              f"nonzero entries: {int((x != 0).sum())}, "
              f"range [{x.min():.3f}, {x.max():.3f}]")
        names = fe.feature_names()
        print("\nFirst 12 features:")
        for i in range(12):
            print(f"  [{i:3d}] {names[i]:<28} = {x[i]:.3f}")
