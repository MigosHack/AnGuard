"""Validated recording windows and aggregate matrices, shared by training workflows."""
import hashlib
import numpy as np
from core import SCHEMA, features, validate_frames
from recordings import records

def windows(paths):
    sessions, seen = {}, set()
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
                fingerprint = hashlib.sha256(player.encode() + expanded[:, :8].astype('<f4').tobytes()).digest()
                if fingerprint in seen:
                    raise ValueError('Duplicate player window; remove duplicated inputs')
                seen.add(fingerprint)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
            yield r, expanded, raw


def load(paths):
    rows, labels, players = [], [], []
    digest = hashlib.sha256()
    for r, expanded, raw in windows(paths):
        rows.append(features(expanded))
        labels.append(r['label'])
        players.append(r['player_id'])
        digest.update(raw)
    if not rows:
        raise ValueError('No recorded windows found')
    return np.asarray(rows), np.asarray(labels), np.asarray(players), digest.hexdigest()
