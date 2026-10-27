import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from advanced import train_dataset
from core import SCHEMA, COLUMNS, AGGREGATES, Model, features
from dataio import load
from dataset import PARTS, prepare, read_dataset, sha256
from forest import export_trees, score_trees


def xor_recordings(path, count=20):
    rng = np.random.default_rng(82)
    with path.open('w') as stream:
        for player in range(count):
            for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                label = int(a == b)
                x = np.zeros((40, 8))
                x[:, :2] = rng.normal((a * 2, b * 2), .03, (40, 2))
                stream.write(json.dumps({'schema': 1, 'player_id': f'p-{player}', 'session_id': f'p-{player}-{label}',
                                         'label': label, 'frames': x.tolist()}) + '\n')


class AdvancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'recordings.jsonl'
        xor_recordings(cls.source)
        cls.directory = cls.root / 'dataset'
        cls.manifest = prepare([cls.source], cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_dataset_is_disjoint_complete_and_deterministic(self):
        parts, manifest = read_dataset(self.directory)
        players, sessions = set(), set()
        for part in PARTS:
            declared = manifest['splits'][part]
            self.assertEqual(set(parts[part][1]), {0, 1})
            self.assertFalse(players & set(declared['players']))
            self.assertFalse(sessions & set(declared['sessions']))
            players.update(declared['players']); sessions.update(declared['sessions'])
            self.assertEqual(declared['sha256'], sha256(self.directory / declared['file']))
        self.assertEqual(len(players), 20)
        second = prepare([self.source], self.root / 'dataset-second')
        self.assertEqual(manifest['splits'], second['splits'])
        with self.assertRaises(FileExistsError):
            prepare([self.source], self.directory)

    def test_modified_dataset_is_rejected_and_leakage_rejected_even_with_updated_checksum(self):
        import shutil
        directory = self.root / 'tampered'
        shutil.copytree(self.directory, directory)
        path = directory / 'test.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            read_dataset(directory)
        path.write_text((directory / 'train.jsonl').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['splits']['test']['sha256'] = sha256(path)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'across'):
            read_dataset(directory)

    def test_small_dataset_is_not_faked_into_four_independent_partitions(self):
        source = self.root / 'small.jsonl'
        xor_recordings(source, 2)
        with self.assertRaisesRegex(ValueError, 'four players'):
            prepare([source], self.root / 'not-created')
        self.assertFalse((self.root / 'not-created').exists())

    def test_portable_forests_match_sklearn_and_reject_cycles(self):
        x, y, _, _ = load([self.source])
        for kind, cls in (('random-forest-v1', RandomForestClassifier), ('extra-trees-v1', ExtraTreesClassifier)):
            estimator = cls(n_estimators=9, max_depth=5, random_state=5).fit(x, y)
            artifact = {'schema': SCHEMA, 'columns': list(COLUMNS), 'aggregates': list(AGGREGATES),
                        'kind': kind, 'trees': export_trees(estimator), 'threshold': .8}
            path = self.root / f'{kind}.json'
            path.write_text(json.dumps(artifact))
            model = Model(path)
            np.testing.assert_allclose([model.score_features(row) for row in x], estimator.predict_proba(x)[:, 1], atol=1e-12)
            bad = copy.deepcopy(artifact)
            bad['trees'][0]['left'][0] = 0
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'cyclic'):
                Model(path)

    def test_float32_input_is_compared_to_double_precision_tree_threshold(self):
        tree = {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [0, -2, -2],
                'threshold': [.1000000001, -2, -2], 'probability': [.5, 0, 1]}
        values = np.zeros(288)
        values[0] = .1  # becomes 0.10000000149 in float32, greater than the double threshold
        self.assertEqual(score_trees([tree], values), 1)

    def test_test_labels_cannot_influence_model_selection_or_calibration(self):
        import shutil
        directory = self.root / 'reversed-test'
        shutil.copytree(self.directory, directory)
        source = directory / 'test.jsonl'
        rows = [json.loads(raw) for raw in source.read_text().splitlines()]
        for row in rows:
            row['label'] = 1 - row['label']
        source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['splits']['test']['sha256'] = sha256(source)
        manifest['splits']['test']['legit_windows'] = sum(row['label'] == 0 for row in rows)
        manifest['splits']['test']['cheat_windows'] = sum(row['label'] == 1 for row in rows)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        first = train_dataset(self.directory, self.root / 'original-test-model.json')
        second = train_dataset(directory, self.root / 'reversed-test-model.json')
        self.assertEqual(first['selected_model'], second['selected_model'])
        self.assertEqual(first['threshold'], second['threshold'])
        self.assertAlmostEqual(first['test']['roc_auc'] + second['test']['roc_auc'], 1)

    def test_forest_runs_through_authenticated_binary_api(self):
        import http.client
        import threading
        from server import InferenceServer
        from test_pipeline import encode
        path = self.root / 'api-forest.json'
        train_dataset(self.directory, path, model_type='random-forest')
        model = Model(path)
        frames = json.loads(self.source.read_text().splitlines()[0])['frames']
        server = InferenceServer(('127.0.0.1', 0), 'test-key', model)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request('POST', '/predict_binary', encode(frames),
                               {'Content-Type': 'application/octet-stream', 'X-Secret-Key': 'test-key'})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertTrue(payload['ready'])
            self.assertAlmostEqual(payload['confidence'], model.score(frames) * 100)
            self.assertAlmostEqual(payload['threshold'], model.threshold * 100)
            connection.close()
        finally:
            server.shutdown(); server.server_close(); worker.join()

    def test_selects_nonlinear_model_on_xor_without_testing_every_candidate_on_test(self):
        output = self.root / 'selected.json'
        report = train_dataset(self.directory, output)
        self.assertIn(report['selected_model'], ('random-forest', 'extra-trees'))
        self.assertEqual(report['selection_partition'], 'validation')
        self.assertEqual(report['threshold_source'], 'calibration_players')
        self.assertEqual(set(report['candidates']), {'logistic', 'random-forest', 'extra-trees'})
        self.assertTrue(all('test' not in value for value in report['candidates'].values()))
        self.assertGreater(report['test']['roc_auc'], .9)
        self.assertEqual(report['deployment_recommendation'], 'observation_only')
        self.assertTrue(output.exists())
        self.assertEqual(json.loads(output.with_name(output.stem + '.report.json').read_text()), report)
        self.assertIn('Held-out test', output.with_name(output.stem + '.report.md').read_text())
        x, y, players, _ = load([self.source])
        model = Model(output)
        self.assertGreater(np.mean([model.score_features(row) for row in x[y == 1]]),
                           np.mean([model.score_features(row) for row in x[y == 0]]))
        with self.assertRaises(FileExistsError):
            train_dataset(self.directory, output)
