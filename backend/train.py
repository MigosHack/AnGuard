"""Train a baseline; calibrate and evaluate on separate players when the dataset permits it."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from core import SCHEMA, COLUMNS, AGGREGATES, features, validate_frames
from recordings import records


def load(paths):
    rows, labels, players, sessions = [], [], [], {}
    digest = hashlib.sha256()
    seen = set()
    for path in paths:
        for number, r, raw in records(path):
            try:
                if r['schema'] not in (1, SCHEMA) or type(r['label']) is not int or r['label'] not in (0, 1):
                    raise ValueError('Invalid schema or label')
                player, session = r['player_id'], r['session_id']
                if not isinstance(player, str) or not player or not isinstance(session, str) or not session:
                    raise ValueError('Missing player/session')
                identity = (player, r['label'])
                if session in sessions and sessions[session] != identity:
                    raise ValueError('Session has inconsistent player or label')
                sessions[session] = identity
                expanded = validate_frames(r['frames'], r['schema'])
                values = features(expanded)
                # Duplicated files/windows must not inflate metrics or training weights.
                fingerprint = hashlib.sha256(player.encode() + expanded[:, :8].astype('<f4').tobytes()).digest()
                if fingerprint in seen:
                    raise ValueError('Duplicate player window; remove duplicated inputs')
                seen.add(fingerprint)
                rows.append(values)
                labels.append(r['label'])
                players.append(player)
                digest.update(raw)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
    if not rows:
        raise ValueError('No recorded windows found')
    return np.asarray(rows), np.asarray(labels), np.asarray(players), digest.hexdigest()


def split_players(x, y, groups, seed):
    train, rest = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed).split(x, y, groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed + 1)
                .split(x[rest], y[rest], groups[rest]))
    calibration, test = rest[a], rest[b]
    for name, indices in [('train', train), ('calibration', calibration), ('test', test)]:
        if set(y[indices]) != {0, 1}:
            raise ValueError(f'{name} split lacks a class; collect more diverse players, do not tune seed on test results')
    return train, calibration, test


def calibrate(scores, labels, groups, target_fpr):
    # Any false alarm during a player's legit calibration sessions counts against that player.
    by_player = {}
    for score, label, player in zip(scores, labels, groups):
        if label == 0:
            by_player[player] = max(float(score), by_player.get(player, 0.0))
    maxima = sorted(by_player.values())
    allowed = math.floor(len(maxima) * target_fpr)
    return float(maxima[len(maxima) - allowed - 1])  # decision uses strict >


def metrics(scores, labels, groups, threshold):
    decisions = scores > threshold
    legit_groups = set(groups[labels == 0])
    cheat_groups = set(groups[labels == 1])
    fp_players = len(set(groups[(labels == 0) & decisions]))
    tp_players = len(set(groups[(labels == 1) & decisions]))
    n = len(legit_groups)
    return {'windows': len(labels), 'players': len(set(groups)),
            'legit_players': n, 'cheat_players': len(cheat_groups),
            'false_alarm_players': fp_players, 'detected_cheat_players': tp_players,
            'player_false_alarm_rate': fp_players / n,
            'player_cheat_recall': tp_players / len(cheat_groups),
            'window_false_positive_rate': float(decisions[labels == 0].mean()),
            'window_recall': float(decisions[labels == 1].mean()),
            'zero_fp_one_sided_95pct_upper_bound': 1 - .05 ** (1 / n) if fp_players == 0 else None}


def train(paths, output, seed=42, target_fpr=.001):
    if not 0 <= target_fpr < 1:
        raise ValueError('target-fpr must be in [0, 1)')
    x, y, groups, digest = load(paths)
    if set(y) != {0, 1}:
        raise ValueError('Training requires both legit (0) and cheat (1) recordings')
    split_warning = None
    try:
        fitting, calibration, test = split_players(x, y, groups, seed)
    except ValueError as error:
        # Small/unsplittable datasets are useful for checking the pipeline, not independent evaluation.
        fitting = np.arange(len(y))
        calibration = test = np.array([], dtype=int)
        split_warning = str(error)
    independent = bool(len(calibration) and len(test))
    scaler = StandardScaler().fit(x[fitting])
    # Each player's recordings have equal total training weight.
    _, inverse, counts = np.unique(groups[fitting], return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / weights.sum()
    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)
    model.fit(scaler.transform(x[fitting]), y[fitting], sample_weight=weights)
    scores = model.predict_proba(scaler.transform(x))[:, 1]
    threshold_data = calibration if independent else fitting
    threshold = calibrate(scores[threshold_data], y[threshold_data], groups[threshold_data], target_fpr)
    training_metrics = metrics(scores[fitting], y[fitting], groups[fitting], threshold)
    training_metrics['zero_fp_one_sided_95pct_upper_bound'] = None
    report = {'validation_mode': 'player_holdout' if independent else 'training_only',
              'validation_unavailable_reason': split_warning,
              'threshold_source': 'calibration_players' if independent else 'training_players',
              'seed': seed, 'dataset_sha256': digest, 'target_calibration_player_fpr': target_fpr if independent else None,
              'target_training_player_fpr': None if independent else target_fpr,
              'threshold': threshold, 'splits': {name: sorted(set(groups[idx])) for name, idx in
                  [('train', fitting), ('calibration', calibration), ('test', test)]},
              'training': training_metrics,
              'calibration': metrics(scores[calibration], y[calibration], groups[calibration], threshold) if independent else None,
              'test': metrics(scores[test], y[test], groups[test], threshold) if independent else None,
              'warning': 'Baseline score is not a calibrated probability. Synthetic tests do not establish game accuracy. '
                         'Zero observed false alarms does not prove zero risk. Training-only scores are not independent evaluation. '
                         'Keep observation mode enabled.'}
    artifact = {'schema': SCHEMA, 'kind': 'standardized-logistic-v2', 'columns': list(COLUMNS), 'aggregates': list(AGGREGATES), 'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(), 'weights': model.coef_[0].tolist(),
                'bias': float(model.intercept_[0]), 'threshold': threshold,
                'dataset_sha256': digest, 'report': report}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting an active model or its evaluation by accident.
    with output.open('x') as stream:
        json.dump(artifact, stream, allow_nan=False, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', nargs='+', help='Recorded JSONL or label-first CSV files (use one format per recording)')
    parser.add_argument('--output', required=True, help='New versioned JSON model file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-fpr', type=float, default=.001)
    args = parser.parse_args()
    try:
        print(json.dumps(train(args.data, args.output, args.seed, args.target_fpr), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f'Training failed: {error}\n')
