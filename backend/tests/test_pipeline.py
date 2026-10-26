import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

import flatbuffers
import numpy as np

from core import Model, decode_flatbuffer, features
from server import InferenceServer
from train import calibrate, load, metrics, train


def encode(frames):
    b = flatbuffers.Builder(4096)
    offsets = []
    for row in frames:
        b.StartObject(8)
        for i, value in enumerate(row):
            b.PrependFloat32Slot(i, value, 0.0)
        offsets.append(b.EndObject())
    b.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        b.PrependUOffsetTRelative(offset)
    vector = b.EndVector()
    b.StartObject(1)
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
        np.testing.assert_allclose(decode_flatbuffer(encode(x)), x, atol=1e-5)

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
            server.model = self.model
            status, result = request('POST', '/predict_binary', body)
            self.assertEqual(status, 200)
            self.assertTrue(result['ready'])
            self.assertFalse(result['suspicious'])
            status, result = request('POST', '/predict_binary', encode(np.full((40, 8), 3)))
            self.assertEqual(status, 200)
            self.assertTrue(result['suspicious'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    @unittest.skipUnless(os.environ.get('ANGUARD_JAVA_FIXTURE'), 'Java fixture not requested')
    def test_existing_java_serializer_compatibility(self):
        data = Path(os.environ['ANGUARD_JAVA_FIXTURE']).read_bytes()
        expected = np.array([[i / 10, -i / 10, .2, -.3, .4, -.5, 0, .001] for i in range(40)])
        np.testing.assert_allclose(decode_flatbuffer(data), expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
