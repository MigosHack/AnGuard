"""Versioned movement schema shared by training and inference."""
import json
import math
import struct
from pathlib import Path

import numpy as np

SCHEMA = 2
FRAMES = 40
BASE_COLUMNS = ('delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch',
                'jerk_pitch', 'jerk_yaw', 'gcd_error_yaw', 'gcd_error_pitch')
INSTANT_COLUMNS = ('abs_delta_yaw', 'abs_delta_pitch', 'rotation_magnitude',
                   'abs_accel_yaw', 'abs_accel_pitch', 'acceleration_magnitude',
                   'abs_jerk_yaw', 'abs_jerk_pitch', 'jerk_magnitude',
                   'yaw_direction_fraction', 'pitch_direction_fraction',
                   'yaw_direction_reversal', 'pitch_direction_reversal',
                   'signed_yaw_change', 'signed_pitch_change', 'rotation_magnitude_change')
ROLLING_BASE = (0, 1, 2, 3, 6, 7)
COLUMNS = BASE_COLUMNS + INSTANT_COLUMNS + tuple(
    f'{BASE_COLUMNS[column]}_rolling20_{stat}'
    for column in ROLLING_BASE for stat in ('mean', 'std', 'min', 'max'))
AGGREGATES = ('mean', 'std', 'p10', 'median', 'p90', 'abs_max')
MODEL_INPUTS = len(COLUMNS) * len(AGGREGATES)


def expand_frames(value):
    """Exactly 48 causal features per sample; only the current/past rows of this window are used."""
    x = np.asarray(value, dtype=np.float64)
    if x.shape != (FRAMES, 8) or not np.isfinite(x).all() or np.max(np.abs(x)) > 1e6:
        raise ValueError('Expected 40 finite frames with eight bounded base columns')
    # Match the Java/FlatBuffers float32 source before deriving features.
    x = x.astype(np.float32).astype(np.float64)
    result = np.zeros((FRAMES, len(COLUMNS)), dtype=np.float64)
    result[:, :8] = x
    previous = np.vstack((x[:1], x[:-1]))
    magnitude = np.hypot(x[:, 0], x[:, 1])
    result[:, 8:24] = np.column_stack((
        np.abs(x[:, 0]), np.abs(x[:, 1]), magnitude,
        np.abs(x[:, 2]), np.abs(x[:, 3]), np.hypot(x[:, 2], x[:, 3]),
        np.abs(x[:, 5]), np.abs(x[:, 4]), np.hypot(x[:, 4], x[:, 5]),
        np.divide(x[:, 0], magnitude, out=np.zeros(FRAMES), where=magnitude != 0),
        np.divide(x[:, 1], magnitude, out=np.zeros(FRAMES), where=magnitude != 0),
        x[:, 0] * previous[:, 0] < 0, x[:, 1] * previous[:, 1] < 0,
        x[:, 0] - previous[:, 0], x[:, 1] - previous[:, 1],
        magnitude - np.hypot(previous[:, 0], previous[:, 1])))
    padded = np.pad(x[:, ROLLING_BASE], ((19, 0), (0, 0)), constant_values=np.nan)
    history = np.lib.stride_tricks.sliding_window_view(padded, 20, axis=0)
    result[:, 24:] = np.stack((np.nanmean(history, axis=2), np.nanstd(history, axis=2),
                              np.nanmin(history, axis=2), np.nanmax(history, axis=2)), axis=2).reshape(FRAMES, 24)
    return result.astype(np.float32).astype(np.float64)


def validate_frames(value, schema=None):
    x = np.asarray(value, dtype=np.float64)
    if schema not in (None, 1, SCHEMA):
        raise ValueError('Unsupported recording schema')
    expected = 8 if schema == 1 else len(COLUMNS) if schema == SCHEMA else None
    if x.shape not in ((FRAMES, 8), (FRAMES, len(COLUMNS))) or (expected and x.shape[1] != expected):
        raise ValueError('Expected 40 frames with 8 legacy or 48 v2 columns')
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1e9:
        raise ValueError('Movement values out of bounds')
    expanded = expand_frames(x[:, :8])
    if x.shape[1] == len(COLUMNS) and not np.allclose(x, expanded, rtol=1e-5, atol=1e-6):
        raise ValueError('Derived features do not match the base movement window')
    return expanded


def features(frames):
    x = validate_frames(frames)
    return np.concatenate((x.mean(0), x.std(0), np.quantile(x, .1, axis=0),
                           np.median(x, axis=0), np.quantile(x, .9, axis=0),
                           np.abs(x).max(0)))


def decode_flatbuffer(data):
    """Strict bounded reader for the existing TickDataSequence schema, not arbitrary buffers."""
    if not 8 <= len(data) <= 65536:
        raise ValueError('Invalid FlatBuffer size')

    def read(fmt, offset):
        size = struct.calcsize('<' + fmt)
        if offset < 0 or offset + size > len(data):
            raise ValueError('Truncated FlatBuffer')
        return struct.unpack_from('<' + fmt, data, offset)[0]

    def field(table, index, width=4):
        vtable = table - read('i', table)
        size = read('H', vtable)
        object_size = read('H', vtable + 2)
        if size < 4 or size % 2 or object_size < 4:
            raise ValueError('Invalid FlatBuffer table')
        read('B', vtable + size - 1)
        read('B', table + object_size - 1)
        slot = 4 + index * 2
        offset = read('H', vtable + slot) if slot + 2 <= size else 0
        if offset and (offset < 4 or offset + width > object_size):
            raise ValueError('Invalid FlatBuffer field')
        return table + offset if offset else None

    root = read('I', 0)
    schema_field = field(root, 1, width=2)
    schema = read('H', schema_field) if schema_field is not None else 1
    if schema not in (1, SCHEMA):
        raise ValueError('Unsupported FlatBuffer schema')
    vector_field = field(root, 0)
    if vector_field is None:
        raise ValueError('Missing movement vector')
    vector = vector_field + read('I', vector_field)
    if read('I', vector) != FRAMES:
        raise ValueError('Expected exactly 40 frames')
    result = []
    for i in range(FRAMES):
        entry = vector + 4 + i * 4
        table = entry + read('I', entry)
        row = []
        for column in range(8):
            offset = field(table, column)
            row.append(read('f', offset) if offset is not None else 0.0)
        if schema == SCHEMA:
            derived_field = field(table, 8)
            if derived_field is None:
                raise ValueError('Missing v2 derived feature vector')
            derived = derived_field + read('I', derived_field)
            if read('I', derived) != 40:
                raise ValueError('Expected 40 derived features per frame')
            row.extend(read('f', derived + 4 + j * 4) for j in range(40))
        result.append(row)
    return validate_frames(result, schema)


class Model:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        d = self.data
        if d.get('schema') != SCHEMA or d.get('kind') != 'standardized-logistic-v2':
            raise ValueError('Unsupported model schema: retrain old models with the 48-column pipeline')
        if d.get('columns') != list(COLUMNS) or d.get('aggregates') != list(AGGREGATES):
            raise ValueError('Model feature order does not match this backend')
        self.mean, self.scale, self.weights = [np.asarray(d[k], dtype=float)
                                               for k in ('mean', 'scale', 'weights')]
        if any(a.shape != (MODEL_INPUTS,) or not np.isfinite(a).all()
               for a in (self.mean, self.scale, self.weights)) or (self.scale <= 0).any():
            raise ValueError('Invalid model parameters')
        self.bias = float(d['bias'])
        self.threshold = float(d['threshold'])
        if not math.isfinite(self.bias) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid model threshold or bias')

    def score(self, frames):
        z = float(((features(frames) - self.mean) / self.scale) @ self.weights + self.bias)
        return float(1 / (1 + math.exp(-max(-700, min(700, z)))))
