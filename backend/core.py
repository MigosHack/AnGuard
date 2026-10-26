"""Versioned movement schema shared by training and inference."""
import json
import math
import struct
from pathlib import Path

import numpy as np

SCHEMA = 1
FRAMES = 40
COLUMNS = ('delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch',
           'jerk_pitch', 'jerk_yaw', 'gcd_error_yaw', 'gcd_error_pitch')


def validate_frames(value):
    x = np.asarray(value, dtype=np.float64)
    if x.shape != (FRAMES, len(COLUMNS)) or not np.isfinite(x).all():
        raise ValueError('Expected 40 finite frames with eight columns')
    if np.max(np.abs(x)) > 1e6:
        raise ValueError('Movement values out of bounds')
    return x


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

    def field(table, index):
        vtable = table - read('i', table)
        size = read('H', vtable)
        object_size = read('H', vtable + 2)
        if size < 4 or size % 2 or object_size < 4:
            raise ValueError('Invalid FlatBuffer table')
        read('B', vtable + size - 1)
        read('B', table + object_size - 1)
        slot = 4 + index * 2
        offset = read('H', vtable + slot) if slot + 2 <= size else 0
        if offset and (offset < 4 or offset + 4 > object_size):
            raise ValueError('Invalid FlatBuffer field')
        return table + offset if offset else None

    root = read('I', 0)
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
        result.append(row)
    return validate_frames(result)


class Model:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        d = self.data
        if d.get('schema') != SCHEMA or d.get('kind') != 'standardized-logistic-v1':
            raise ValueError('Unsupported model schema')
        self.mean, self.scale, self.weights = [np.asarray(d[k], dtype=float)
                                               for k in ('mean', 'scale', 'weights')]
        if any(a.shape != (48,) or not np.isfinite(a).all()
               for a in (self.mean, self.scale, self.weights)) or (self.scale <= 0).any():
            raise ValueError('Invalid model parameters')
        self.bias = float(d['bias'])
        self.threshold = float(d['threshold'])
        if not math.isfinite(self.bias) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid model threshold or bias')

    def score(self, frames):
        z = float(((features(frames) - self.mean) / self.scale) @ self.weights + self.bias)
        return float(1 / (1 + math.exp(-max(-700, min(700, z)))))
