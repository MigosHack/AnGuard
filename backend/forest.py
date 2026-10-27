"""Portable JSON forest export/inference; no executable pickle artifacts."""
import numpy as np

from core import MODEL_INPUTS

KINDS = ('random-forest-v1', 'extra-trees-v1')


def export_trees(estimator):
    result = []
    for model in estimator.estimators_:
        tree = model.tree_
        counts = tree.value[:, 0, :]
        probability = counts[:, 1] / counts.sum(axis=1)
        result.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
                       'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
                       'probability': probability.tolist()})
    return result


def validate_trees(trees):
    if not isinstance(trees, list) or not 1 <= len(trees) <= 512:
        raise ValueError('Invalid forest size')
    total = 0
    normalized = []
    for tree in trees:
        arrays = {name: np.asarray(tree[name], dtype=float) for name in ('left', 'right', 'feature', 'threshold', 'probability')}
        n = len(arrays['left'])
        total += n
        if n < 1 or total > 1_000_000 or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Invalid forest arrays')
        left, right, feature = (arrays[key] for key in ('left', 'right', 'feature'))
        if any(not np.equal(a, np.floor(a)).all() for a in (left, right, feature)):
            raise ValueError('Invalid noninteger tree indices')
        probability = arrays['probability']
        if (probability < 0).any() or (probability > 1).any():
            raise ValueError('Invalid tree probability')
        parents = np.zeros(n, dtype=int)
        for node in range(n):
            if left[node] == -1 and right[node] == -1:
                continue
            if not 0 <= feature[node] < MODEL_INPUTS:
                raise ValueError('Invalid tree feature index')
            for child in (left[node], right[node]):
                if not node < child < n:
                    raise ValueError('Invalid or cyclic tree child')
                parents[int(child)] += 1
        if parents[0] != 0 or (parents[1:] != 1).any():
            raise ValueError('Tree must be connected without shared children')
        normalized.append({name: array.astype(int).tolist() if name in ('left', 'right', 'feature') else array.tolist()
                           for name, array in arrays.items()})
    return normalized


def score_trees(trees, values):
    # sklearn's forest converts its input matrix to float32 before splitting.
    values = np.asarray(values, dtype=np.float32)
    result = 0.0
    for tree in trees:
        node = 0
        while tree['left'][node] != -1:
            node = tree['left'][node] if float(values[tree['feature'][node]]) <= tree['threshold'][node] else tree['right'][node]
        result += tree['probability'][node]
    return float(result / len(trees))
