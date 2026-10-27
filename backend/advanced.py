"""Select on validation, set threshold on calibration, then evaluate the selected model on test."""
import json
from pathlib import Path
import time
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from core import SCHEMA, COLUMNS, AGGREGATES, Model
from dataset import read_dataset, sha256, PARTS
from forest import export_trees


def player_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * len(weights) / weights.sum()


def evaluate(scores, labels, players, threshold, independent=False):
    from train import metrics
    result = metrics(scores, labels, players, threshold)
    if not independent:
        result['zero_fp_one_sided_95pct_upper_bound'] = None
    weights = player_weights(players)
    result.update({'roc_auc': float(roc_auc_score(labels, scores, sample_weight=weights)),
                   'average_precision': float(average_precision_score(labels, scores, sample_weight=weights)),
                   'partial_roc_auc_at_1pct_fpr': float(roc_auc_score(labels, scores, max_fpr=.01, sample_weight=weights))})
    tn, fp, fn, tp = confusion_matrix(labels, scores > threshold, labels=[0, 1]).ravel()
    result['confusion_windows'] = {name: int(value) for name, value in zip(('tn', 'fp', 'fn', 'tp'), (tn, fp, fn, tp))}
    result['window_precision'] = float(tp / (tp + fp)) if tp + fp else 0.0
    result['score_quantiles'] = {name: dict(zip(('min', 'median', 'p95', 'max'), map(float, np.quantile(scores[labels == label], [0, .5, .95, 1]))))
                                 for name, label in (('legit', 0), ('cheat', 1))}
    return result


def candidates(seed, selection):
    choices = {
        'logistic': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed),
        'random-forest': RandomForestClassifier(n_estimators=96, max_depth=14, max_leaf_nodes=2048, min_samples_leaf=4,
                                                class_weight='balanced', max_features='sqrt', n_jobs=2, random_state=seed),
        'extra-trees': ExtraTreesClassifier(n_estimators=96, max_depth=14, max_leaf_nodes=2048, min_samples_leaf=4,
                                           class_weight='balanced', max_features='sqrt', n_jobs=2, random_state=seed),
    }
    if selection not in ('auto', *choices):
        raise ValueError('Unknown candidate model')
    return choices if selection == 'auto' else {selection: choices[selection]}


def markdown_report(report):
    lines = ['# SilentAC: training report', '',
             f"Selected model: **{report['selected_model']}**. Threshold: **{report['threshold']:.6f}**.",
             'Model selection: validation players. Threshold selection: calibration players. Final evaluation: test players.', '',
             '| Model | Validation player recall | Validation window recall | Validation ROC-AUC |',
             '|---|---:|---:|---:|']
    for name, candidate in report['candidates'].items():
        r = candidate['validation']
        lines.append(f"| {name} | {r['player_cheat_recall']:.4f} | {r['window_recall']:.4f} | {r['roc_auc']:.4f} |")
    r = report['test']
    lines += ['', '## Held-out test', '',
              f"Players: {r['players']} ({r['legit_players']} with legit, {r['cheat_players']} with cheat recordings). Windows: {r['windows']}.",
              f"Player false-alarm rate: {r['player_false_alarm_rate']:.6f}. Player cheat recall: {r['player_cheat_recall']:.6f}.",
              f"Window precision: {r['window_precision']:.6f}. Window recall: {r['window_recall']:.6f}.",
              f"Confusion matrix (windows): {r['confusion_windows']}.",
              f"ROC-AUC: {r['roc_auc']:.6f}. Average precision: {r['average_precision']:.6f}."]
    upper = r['zero_fp_one_sided_95pct_upper_bound']
    if upper is not None:
        lines.append(f"Zero observed false-alarm players: one-sided 95% upper bound **{upper:.6f}**, under independent representative-player assumptions.")
    lines += ['', '**Deployment recommendation: observation only.**', '', report['warning'], '',
              'Feature importances are descriptive and can be unstable when features are correlated. They do not prove a causal cheat signal.', '']
    return '\n'.join(lines)


def train_dataset(directory, output, seed=42, target_fpr=.001, model_type='auto'):
    from train import calibrate
    if not 0 <= target_fpr < 1:
        raise ValueError('target-fpr must be in [0, 1)')
    output = Path(output)
    report_json = output.with_name(output.stem + '.report.json')
    report_md = output.with_name(output.stem + '.report.md')
    for path in (output, report_json, report_md):
        if path.exists():
            raise FileExistsError(f'Output already exists: {path}')
    partitions, manifest = read_dataset(directory)
    print("Loaded frozen dataset: " + ", ".join(f"{part}={len(partitions[part][1])} windows" for part in PARTS), file=sys.stderr, flush=True)
    x_train, y_train, g_train = partitions['train']
    x_val, y_val, g_val = partitions['validation']
    comparisons, fitted = {}, {}
    with threadpool_limits(limits=2):
        for name, estimator in candidates(seed, model_type).items():
            print(f"Fitting {name} on train; comparing on validation...", file=sys.stderr, flush=True)
            started = time.perf_counter()
            scaler = StandardScaler().fit(x_train) if name == 'logistic' else None
            fit_x = scaler.transform(x_train) if scaler else x_train
            estimator.fit(fit_x, y_train, sample_weight=player_weights(g_train))
            val_scores = estimator.predict_proba(scaler.transform(x_val) if scaler else x_val)[:, 1]
            provisional = calibrate(val_scores, y_val, g_val, target_fpr)
            validation = evaluate(val_scores, y_val, g_val, provisional)
            comparisons[name] = {'validation': validation, 'provisional_validation_threshold': provisional,
                                 'fit_seconds': time.perf_counter() - started, 'parameters': estimator.get_params()}
            fitted[name] = estimator, scaler
    def rank(name):
        r = comparisons[name]['validation']
        return (r['player_cheat_recall'], r['window_recall'], r['partial_roc_auc_at_1pct_fpr'], r['roc_auc'])
    selected = max(comparisons, key=rank)
    print(f"Selected {selected}; setting threshold on calibration, then evaluating test...", file=sys.stderr, flush=True)
    estimator, scaler = fitted[selected]
    # Selection is complete before calibration/test scores are computed.
    def score(part):
        x, y, groups = partitions[part]
        return estimator.predict_proba(scaler.transform(x) if scaler else x)[:, 1]
    cal_scores = score('calibration')
    _, cal_y, cal_groups = partitions['calibration']
    threshold = calibrate(cal_scores, cal_y, cal_groups, target_fpr)
    test_scores = score('test')
    test_x, test_y, test_groups = partitions['test']
    report = {'workflow': 'four-way-model-selection-v1', 'validation_mode': 'player_holdout',
              'selected_model': selected, 'selection_partition': 'validation',
              'selection_rule': 'player recall, window recall, partial ROC-AUC, ROC-AUC at provisional validation player-FPR threshold',
              'threshold_source': 'calibration_players', 'threshold': threshold,
              'target_calibration_player_fpr': target_fpr, 'dataset_sha256': manifest['source_sha256'],
              'dataset_manifest_sha256': sha256(Path(directory) / 'manifest.json'), 'seed': seed,
              'splits': {part: manifest['splits'][part]['players'] for part in PARTS},
              'partition_counts': {part: {key: manifest['splits'][part][key] for key in ('windows', 'legit_windows', 'cheat_windows')} for part in PARTS},
              'candidates': comparisons,
              'calibration': evaluate(cal_scores, cal_y, cal_groups, threshold),
              'test': evaluate(test_scores, test_y, test_groups, threshold, independent=True),
              'deployment_recommendation': 'observation_only',
              'warning': 'Scores are not calibrated probabilities of cheating. Small held-out cohorts cannot establish a low false-alarm rate. '
                         'Dataset/model selection on validation does not replace live whole-plugin evaluation. Do not tune on test results.'}
    importance = np.abs(estimator.coef_[0]) if scaler else estimator.feature_importances_
    names = [f'{column}:{aggregate}' for aggregate in AGGREGATES for column in COLUMNS]
    report['top_features'] = [{'feature': names[i], 'importance': float(importance[i])}
                              for i in np.argsort(importance)[-20:][::-1]]
    artifact = {'schema': SCHEMA, 'columns': list(COLUMNS), 'aggregates': list(AGGREGATES),
                'threshold': threshold, 'dataset_sha256': manifest['source_sha256'], 'report': report}
    if scaler:
        artifact.update(kind='standardized-logistic-v2', mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                        weights=estimator.coef_[0].tolist(), bias=float(estimator.intercept_[0]))
    else:
        artifact.update(kind='random-forest-v1' if selected == 'random-forest' else 'extra-trees-v1', trees=export_trees(estimator))
    # Validate portability against sklearn before publishing the artifact. No pickle loading is involved.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='silentac-export-') as temp:
        check = Path(temp) / 'model.json'
        check.write_text(json.dumps(artifact, allow_nan=False))
        portable = Model(check)
        indices = np.linspace(0, len(test_x) - 1, min(64, len(test_x)), dtype=int)
        np.testing.assert_allclose([portable.score_features(test_x[i]) for i in indices], test_scores[indices], rtol=1e-10, atol=1e-12)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(artifact, stream, allow_nan=False, indent=2)
    with report_json.open('x') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
    with report_md.open('x') as stream:
        stream.write(markdown_report(report))
    return report
