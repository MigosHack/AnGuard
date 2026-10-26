"""Read window JSONL or label-first per-frame CSV without losing player/session boundaries."""
import csv
import json
from pathlib import Path

from core import COLUMNS, FRAMES, SCHEMA

CSV_COLUMNS = ('label', *COLUMNS, 'player_id', 'session_id', 'window_id',
               'frame_index', 'timestamp_ms', 'schema')


def _csv_records(path):
    with path.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError(f'{path}: CSV header must contain label, 48 named features and grouping metadata')
        current, frames, raw, seen = None, [], [], set()
        try:
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Wrong CSV column count')
                metadata = (row['player_id'], row['session_id'], row['window_id'],
                            int(row['label']), int(row['timestamp_ms']), int(row['schema']))
                if metadata[3] not in (0, 1) or metadata[5] != SCHEMA or not all(metadata[:3]):
                    raise ValueError('Invalid CSV label, schema or identity')
                index = int(row['frame_index'])
                if current is None:
                    if metadata[2] in seen:
                        raise ValueError('Repeated CSV window_id')
                    current = metadata
                if metadata != current or index != len(frames):
                    raise ValueError('CSV windows must contain consecutive frames 0..39 with consistent metadata')
                frames.append([float(row[column]) for column in COLUMNS])
                raw.append(json.dumps(row, sort_keys=True).encode())
                if len(frames) == FRAMES:
                    player, session, window, label, timestamp, schema = current
                    yield reader.line_num, {'schema': schema, 'label': label, 'player_id': player,
                                            'session_id': session, 'window_id': window,
                                            'timestamp_ms': timestamp, 'frames': frames}, b''.join(raw)
                    seen.add(window)
                    current, frames, raw = None, [], []
            if frames:
                raise ValueError('Incomplete CSV window: expected 40 frames')
        except (ValueError, KeyError, TypeError, csv.Error) as error:
            raise ValueError(f'{path}:{reader.line_num}: {error}') from error


def records(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        yield from _csv_records(path)
        return
    with path.open('rb') as source:
        for number, raw in enumerate(source, 1):
            try:
                if len(raw) > 131072:
                    raise ValueError('Oversized record')
                yield number, json.loads(raw), raw
            except (ValueError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
