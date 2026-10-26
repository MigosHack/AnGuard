import csv
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

import flatbuffers
import numpy as np

from core import COLUMNS, MODEL_INPUTS, Model, decode_flatbuffer, expand_frames, features, validate_frames
from server import InferenceServer
from train import calibrate, load, metrics, train
from recordings import CSV_COLUMNS, records


def encode(frames, schema=1):
    b = flatbuffers.Builder(4096)
    offsets = []
    for row in frames:
        derived = None
        if schema == 2:
            b.StartVector(4, len(row[8:]), 4)
            for value in reversed(row[8:]):
                b.PrependFloat32(value)
            derived = b.EndVector()
        b.StartObject(9 if schema == 2 else 8)
        if derived is not None:
            b.PrependUOffsetTRelativeSlot(8, derived, 0)
        for i, value in enumerate(row[:8]):
            b.PrependFloat32Slot(i, value, 0.0)
        offsets.append(b.EndObject())
    b.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        b.PrependUOffsetTRelative(offset)
    vector = b.EndVector()
    b.StartObject(2)
    b.PrependUint16Slot(1, schema, 1)
    b.PrependUOffsetTRelativeSlot(0, vector, 0)
    b.Finish(b.EndObject())
    return bytes(b.Output())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = cls.root / 'synthetic.jsonl'
        rng = np.random.default_rng(52)
        # Synthetic fixtures exercise plumbing only, never claim real detection quality.
        with cls.data.open('w') as out:
            for player in range(30):
                for label in (0, 1):
                    for window in range(3):
                        frames = rng.normal(label * 3, .4, (40, 8)).tolist()
                        out.write(json.dumps({'schema': 1, 'player_id': str(player),
                                              'session_id': f'{player}-{label}', 'label': label,
                                              'frames': frames}) + '\n')
        cls.model_path = cls.root / 'baseline.json'
        cls.report = train([cls.data], cls.model_path)
        cls.model = Model(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_flatbuffer_roundtrip_and_omitted_fields(self):
        x = np.arange(320, dtype=float).reshape(40, 8) / 10
        x[:, 4] = 0
        np.testing.assert_allclose(decode_flatbuffer(encode(x)), expand_frames(x), atol=1e-5)

    def test_exactly_48_per_sample_and_288_model_inputs(self):
        result = expand_frames(np.zeros((40, 8)))
        self.assertEqual(len(COLUMNS), 48)
        self.assertEqual(len(set(COLUMNS)), 48)
        self.assertEqual(result.shape, (40, 48))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(MODEL_INPUTS, 288)
        self.assertEqual(features(result).shape, (288,))

    def test_features_are_causal_and_have_known_values(self):
        x = np.zeros((40, 8))
        x[0, :2] = (3, 4)
        x[1, :2] = (-3, -4)
        result = expand_frames(x)
        self.assertEqual(result[0, 10], 5)
        self.assertEqual(result[0, 19], 0)
        self.assertEqual(result[1, 19], 1)
        self.assertEqual(result[1, 20], 1)
        self.assertEqual(result[1, 21], -6)
        self.assertEqual(result[1, 22], -8)
        self.assertEqual(result[1, 24], 0)  # yaw rolling mean
        self.assertEqual(result[1, 25], 3)  # yaw population std
        self.assertEqual(result[1, 26], -3)
        self.assertEqual(result[1, 27], 3)
        x[20:] = 99
        np.testing.assert_array_equal(expand_frames(x)[:20], result[:20])

    def test_v2_binary_roundtrip_and_corrupt_derived_rejected(self):
        x = expand_frames(np.random.default_rng(3).normal(size=(40, 8)))
        np.testing.assert_array_equal(decode_flatbuffer(encode(x, schema=2)), x)
        x[0, 10] += 1
        with self.assertRaisesRegex(ValueError, 'Derived'):
            decode_flatbuffer(encode(x, schema=2))
        with self.assertRaises(ValueError):
            decode_flatbuffer(encode(np.zeros((40, 8)), schema=2))
        with self.assertRaisesRegex(ValueError, 'schema'):
            decode_flatbuffer(encode(np.zeros((40, 8)), schema=3))

    def test_legacy_records_expand_identically_to_v2(self):
        v2 = self.root / 'v2-recordings.jsonl'
        with v2.open('w') as out:
            for raw in self.data.read_text().splitlines():
                row = json.loads(raw)
                row['schema'] = 2
                row['frames'] = expand_frames(row['frames']).tolist()
                out.write(json.dumps(row) + '\n')
        np.testing.assert_array_equal(load([self.data])[0], load([v2])[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load([self.data, v2])

    def test_schema_mismatch_and_old_models_rejected(self):
        with self.assertRaises(ValueError):
            validate_frames(np.zeros((40, 8)), schema=2)
        with self.assertRaises(ValueError):
            validate_frames(np.zeros((40, 48)), schema=1)
        artifact = json.loads(self.model_path.read_text())
        artifact['schema'] = 1
        path = self.root / 'old-model.json'
        path.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'retrain'):
            Model(path)
        artifact['schema'] = 2
        artifact['columns'][0], artifact['columns'][1] = artifact['columns'][1], artifact['columns'][0]
        path.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'order'):
            Model(path)

    def test_malformed_flatbuffers_rejected(self):
        valid = encode(np.zeros((40, 8)))
        for payload in (b'', b'abcdefgh', valid[:20], encode(np.zeros((39, 8))),
                        encode(np.full((40, 8), float('nan'))), b'\xff' * 80):
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                decode_flatbuffer(payload)

    def test_invalid_json_frames_rejected(self):
        for x in ([], [[0] * 8] * 39, [[float('inf')] * 8] * 40):
            with self.assertRaises(ValueError):
                features(x)

    def test_player_disjoint_training_and_evaluation(self):
        split = [set(v) for v in self.report['splits'].values()]
        self.assertEqual(len(set.union(*split)), 30)
        for i in range(3):
            for j in range(i):
                self.assertFalse(split[i] & split[j])
        self.assertGreater(self.report['test']['windows'], 0)
        result = self.report['test']
        if result['false_alarm_players'] == 0:
            self.assertGreater(result['zero_fp_one_sided_95pct_upper_bound'], 0)
        else:
            self.assertIsNone(result['zero_fp_one_sided_95pct_upper_bound'])

    def test_zero_observed_false_alarms_reports_uncertainty(self):
        result = metrics(np.array([.1, .8]), np.array([0, 1]), np.array(['a', 'b']), .5)
        self.assertEqual(result['player_false_alarm_rate'], 0)
        self.assertAlmostEqual(result['zero_fp_one_sided_95pct_upper_bound'], .95)

    def test_model_uses_same_score_as_training(self):
        x = np.zeros((40, 8))
        y = np.full((40, 8), 3)
        self.assertLess(self.model.score(x), self.model.threshold)
        self.assertGreater(self.model.score(y), self.model.threshold)
        self.assertGreater(self.report['test']['player_cheat_recall'], .9)

    def test_threshold_counts_any_false_alarm_per_player(self):
        scores = np.array([.1, .9, .2, .3, .99])
        labels = np.array([0, 0, 0, 0, 1])
        groups = np.array(['a', 'a', 'b', 'b', 'c'])
        self.assertEqual(calibrate(scores, labels, groups, 0), .9)
        self.assertEqual(calibrate(scores, labels, groups, .5), .3)

    def test_training_refuses_duplicate_data(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load([self.data, self.data])

    def test_training_refuses_missing_classes(self):
        source = self.root / 'legit-only.jsonl'
        source.write_text(''.join(raw for raw in self.data.read_text().splitlines(keepends=True)
                                  if json.loads(raw)['label'] == 0))
        with self.assertRaisesRegex(ValueError, 'lacks a class'):
            train([source], self.root / 'must-not-exist.json')
        self.assertFalse((self.root / 'must-not-exist.json').exists())

    def test_training_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            train([self.data], self.model_path)

    def test_invalid_model_rejected(self):
        d = json.loads(self.model_path.read_text())
        d['scale'][0] = 0
        path = self.root / 'bad.json'
        path.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            Model(path)

    def test_http_auth_validation_untrained_and_prediction(self):
        server = InferenceServer(('127.0.0.1', 0), 'test-only-secret-with-24-chars')
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def request(method, path, data=None, key='test-only-secret-with-24-chars'):
            conn = http.client.HTTPConnection(*server.server_address, timeout=5)
            conn.request(method, path, body=data, headers={
                'X-Secret-Key': key, 'Content-Type': 'application/octet-stream'})
            response = conn.getresponse()
            result = response.status, json.loads(response.read())
            conn.close()
            return result
        try:
            self.assertEqual(request('GET', '/health'), (200, {'status': 'ok', 'ready': False}))
            body = encode(np.zeros((40, 8)))
            self.assertEqual(request('POST', '/predict_binary', body, key='wrong')[0], 401)
            self.assertEqual(request('POST', '/predict_binary', b'bad')[0], 413)
            self.assertEqual(request('POST', '/predict_binary', b'abcdefgh')[0], 400)
            self.assertEqual(request('POST', '/predict_binary', body)[0], 503)
            v2body = encode(expand_frames(np.zeros((40, 8))), schema=2)
            self.assertEqual(request('POST', '/predict_binary', v2body)[0], 503)
            server.model = self.model
            status, result = request('POST', '/predict_binary', body)
            self.assertEqual(status, 200)
            self.assertTrue(result['ready'])
            self.assertFalse(result['suspicious'])
            self.assertEqual(request('POST', '/predict_binary', v2body), (status, result))
            status, result = request('POST', '/predict_binary', encode(np.full((40, 8), 3)))
            self.assertEqual(status, 200)
            self.assertTrue(result['suspicious'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def write_csv(self, path):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            for window, raw in enumerate(self.data.read_text().splitlines()):
                r = json.loads(raw)
                for index, frame in enumerate(expand_frames(r['frames'])):
                    writer.writerow([r['label'], *frame, r['player_id'], r['session_id'],
                                     str(window), index, 1000, 2])

    def test_csv_training_matches_jsonl_and_refuses_duplicate_formats(self):
        path = self.root / 'recordings.csv'
        self.write_csv(path)
        json_x, json_y, json_players, _ = load([self.data])
        csv_x, csv_y, csv_players, _ = load([path])
        np.testing.assert_array_equal(csv_x, json_x)
        np.testing.assert_array_equal(csv_y, json_y)
        np.testing.assert_array_equal(csv_players, json_players)
        report = train([path], self.root / 'csv-model.json')
        self.assertEqual(report['test'], self.report['test'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load([self.data, path])

    def test_csv_rejects_missing_metadata_partial_windows_and_mixed_labels(self):
        path = self.root / 'bad.csv'
        path.write_text('label,delta_yaw\n0,1.2\n')
        with self.assertRaisesRegex(ValueError, 'header'):
            list(records(path))
        frames = expand_frames(np.zeros((40, 8)))
        def write(count=40, change=None):
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_COLUMNS)
                for i in range(count):
                    row = [0, *frames[i], 'player', 'session', 'window', i, 1000, 2]
                    if change and i == 1:
                        row[change[0]] = change[1]
                    writer.writerow(row)
        write(count=39)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            list(records(path))
        for change in ((0, 1), (52, 0), (49, 'other-player'), (50, 'other-session'), (54, 1)):
            write(change=change)
            with self.assertRaises(ValueError):
                list(records(path))

    @unittest.skipUnless(os.environ.get('ANGUARD_JAVA_FIXTURE'), 'Java fixture not requested')
    def test_existing_java_serializer_compatibility(self):
        data = Path(os.environ['ANGUARD_JAVA_FIXTURE']).read_bytes()
        expected = np.array([[i / 10, -i / 10, .2, -.3, .4, -.5, 0, .001] for i in range(40)])
        np.testing.assert_allclose(decode_flatbuffer(data), expand_frames(expected), atol=1e-6)
        v2 = Path(os.environ['ANGUARD_JAVA_FIXTURE'] + '.v2').read_bytes()
        java_features = json.loads(Path(os.environ['ANGUARD_JAVA_FIXTURE'] + '.features.json').read_text())
        np.testing.assert_allclose(java_features, expand_frames(np.asarray(java_features)[:, :8]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(decode_flatbuffer(v2), java_features, rtol=1e-6, atol=1e-6)
        csv_window = list(records(Path(os.environ['ANGUARD_JAVA_FIXTURE'] + '.csv')))
        self.assertEqual(len(csv_window), 1)
        self.assertEqual(csv_window[0][1]['label'], 1)
        self.assertEqual(csv_window[0][1]['player_id'], '00000000-0000-0000-0000-000000000001')
        np.testing.assert_allclose(validate_frames(csv_window[0][1]['frames'], schema=2), java_features, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
