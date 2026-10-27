"""Freeze train/validation/calibration/test recordings with disjoint players and file checksums."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from core import SCHEMA, COLUMNS, features
from dataio import windows

PARTS = ('train', 'validation', 'calibration', 'test')
RATIOS = (.55, .15, .15, .15)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assignments(labels_by_player, seed):
    """Class coverage is determined from metadata before any model or scores are inspected."""
    buckets = {0: [], 1: [], 'both': []}
    for player in sorted(labels_by_player):
        labels = labels_by_player[player]
        buckets['both' if len(labels) == 2 else next(iter(labels))].append(player)
    if any(sum(label in labels for labels in labels_by_player.values()) < 4 for label in (0, 1)):
        raise ValueError('Independent four-way dataset requires at least four players with legit recordings and four with cheat recordings. '
                         'The same four players may provide both classes. Quick --train still works with fewer players.')
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result = {part: [] for part in PARTS}
    for part in PARTS:
        if buckets['both']:
            result[part].append(buckets['both'].pop())
        else:
            result[part].extend((buckets[0].pop(), buckets[1].pop()))
    remaining = [p for bucket in buckets.values() for p in bucket]
    rng.shuffle(remaining)
    targets = dict(zip(PARTS, (len(labels_by_player) * ratio for ratio in RATIOS)))
    for player in remaining:
        part = max(PARTS, key=lambda key: targets[key] - len(result[key]))
        result[part].append(player)
    return {part: sorted(players) for part, players in result.items()}


def prepare(paths, output, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Dataset already exists: {output}; use a new versioned directory')
    labels_by_player = defaultdict(set)
    duplicate_patterns = defaultdict(set)
    digest = hashlib.sha256()
    count = 0
    for r, expanded, raw in windows(paths):
        labels_by_player[r['player_id']].add(r['label'])
        pattern = hashlib.sha256(expanded[:, :8].astype('<f4').tobytes()).digest()
        duplicate_patterns[pattern].add(r['player_id'])
        digest.update(raw)
        count += 1
    if not count:
        raise ValueError('No recorded windows found')
    split = assignments(labels_by_player, seed)
    membership = {player: part for part, players in split.items() for player in players}
    if any(len({membership[player] for player in players}) > 1 for players in duplicate_patterns.values()):
        raise ValueError('Identical movement windows occur across different players/partitions; remove copied data before splitting')
    output.mkdir(parents=True, exist_ok=False)
    counts = {part: {'windows': 0, 'legit_windows': 0, 'cheat_windows': 0, 'sessions': set()} for part in PARTS}
    second_digest = hashlib.sha256()
    with ExitStack() as stack:
        files = {part: stack.enter_context((output / f'{part}.jsonl').open('x')) for part in PARTS}
        for r, expanded, raw in windows(paths):
            part = membership[r['player_id']]
            normalized = dict(r, schema=SCHEMA, frames=expanded.tolist())
            files[part].write(json.dumps(normalized, allow_nan=False, separators=(',', ':')) + '\n')
            counts[part]['windows'] += 1
            counts[part]['cheat_windows' if r['label'] else 'legit_windows'] += 1
            counts[part]['sessions'].add(r['session_id'])
            second_digest.update(raw)
    if second_digest.digest() != digest.digest():
        raise ValueError('Input recordings changed during preparation. Stop recording and prepare a new dataset directory.')
    manifest = {'format': 'silentac-dataset-v1', 'schema': SCHEMA, 'columns': list(COLUMNS),
                'seed': seed, 'source_sha256': digest.hexdigest(), 'requested_player_ratios': dict(zip(PARTS, RATIOS)),
                'splits': {}, 'sources': [str(p) for p in paths]}
    for part in PARTS:
        stats = counts[part]
        stats['sessions'] = sorted(stats['sessions'])
        manifest['splits'][part] = dict(stats, players=split[part], file=f'{part}.jsonl', sha256=sha256(output / f'{part}.jsonl'))
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2)
    return manifest


def read_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'silentac-dataset-v1' or manifest.get('schema') != SCHEMA or manifest.get('columns') != list(COLUMNS):
        raise ValueError('Unsupported dataset format or feature schema')
    result, all_players, all_sessions, all_patterns = {}, set(), set(), {}
    for part in PARTS:
        declared = manifest['splits'][part]
        if declared['file'] != f'{part}.jsonl':
            raise ValueError('Dataset split filename does not match its partition')
        path = directory / declared['file']
        if sha256(path) != declared['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {part}; do not modify a frozen dataset')
        x, y, players, sessions = [], [], [], set()
        for r, expanded, raw in windows([path]):
            pattern = hashlib.sha256(expanded[:, :8].astype('<f4').tobytes()).digest()
            if pattern in all_patterns and all_patterns[pattern] != part:
                raise ValueError('Duplicate movement window across dataset partitions')
            all_patterns[pattern] = part
            x.append(features(expanded)); y.append(r['label']); players.append(r['player_id']); sessions.add(r['session_id'])
        unique_players = set(players)
        if set(y) != {0, 1} or not unique_players:
            raise ValueError(f'{part} must contain both legit and cheat recordings')
        if unique_players & all_players or sessions & all_sessions:
            raise ValueError('Player/session leakage across dataset partitions')
        if unique_players != set(declared['players']) or sessions != set(declared['sessions']) or len(y) != declared['windows']:
            raise ValueError(f'{part} metadata does not match manifest')
        if sum(v == 0 for v in y) != declared['legit_windows'] or sum(v == 1 for v in y) != declared['cheat_windows']:
            raise ValueError(f'{part} class counts do not match manifest')
        all_players.update(unique_players); all_sessions.update(sessions)
        result[part] = np.asarray(x), np.asarray(y), np.asarray(players)
    return result, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = prepare(args.data, args.output, args.seed)
        print(json.dumps({'dataset': args.output, 'splits': {p: {k: v for k, v in manifest['splits'][p].items() if k not in ('sessions', 'players')} for p in PARTS}}, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Dataset preparation failed: {error}\n')
