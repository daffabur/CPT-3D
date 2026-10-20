"""Fault and corruption tests using a copy of one already generated benchmark volume."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import v3_24k_writer as writer
from .v3_dataset_production import _read_saved


class WriterRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parents[1] / 'data/synthetic3d_v2/schemagan_v3_final_pilot'
        cls.fixture_root = fixture
        cls.fixture_entry = json.loads((fixture / 'manifest.json').read_text())['entries'][0]
        cls.volume, cls.observed = _read_saved(fixture, cls.fixture_entry)
        cls.planned = {k: cls.fixture_entry[k] for k in (
            'realization_id', 'scenario_id', 'lineage_group', 'split', 'family', 'grid_name',
            'shape', 'spacing_m', 'origin_xy_m', 'geometry_tag', 'response_tag', 'generator_version',
            'n_cpt', 'geology_seed', 'material_seed', 'seed', 'site_seed', 'evaluation_site_seed')}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _fake(self, *args):
        return self.volume, self.observed

    def _first(self, point):
        with patch.object(writer, 'make_volume', self._fake):
            with self.assertRaises(writer.InjectedCrash):
                writer.write_volume(self.root, self.planned, {}, 0, 1, point)

    def _recover(self):
        # Minimal one-entry plan for the recovery protocol; the full frozen plan is
        # separately checked by load_frozen/validate_plan in the benchmark.
        plan = {'entries': [self.planned], 'config_sha256': 'fixture'}
        records, status = writer.resume(self.root, plan, {}, checkpoint_now=True)
        return records, status

    def test_fault_points(self):
        for point in writer.FAULT_POINTS[:5]:
            with self.subTest(point=point):
                with tempfile.TemporaryDirectory() as tmp:
                    self.root = Path(tmp)
                    self._first(point)
                    records, status = self._recover()
                    expected = 0 if point == 'before_rename' else 1
                    self.assertEqual(len(records), expected)
                    if expected == 0:
                        with patch.object(writer, 'make_volume', self._fake):
                            record, _ = writer.write_volume(self.root, self.planned, {}, 0, 1)
                            writer.checkpoint(self.root, {'config_sha256': 'fixture'}, [record])
                    else:
                        self.assertEqual(status['recovered_orphans'] + int(point == 'after_journal'), 1)
                    self.assertTrue((self.root / self.planned['realization_id']).exists())
                    self.assertEqual(len(list(self.root.glob(self.planned['realization_id']))), 1)
                    self.assertTrue(writer.validate_full(self.root, {'entries': [self.planned], 'config_sha256': 'fixture'}, {}, reproduce=False)['valid'])

    def test_checkpoint_fault_points(self):
        plan = {'entries': [self.planned], 'config_sha256': 'fixture'}
        for point in writer.FAULT_POINTS[5:]:
            with self.subTest(point=point):
                with tempfile.TemporaryDirectory() as tmp:
                    self.root = Path(tmp)
                    with patch.object(writer, 'make_volume', self._fake):
                        record, _ = writer.write_volume(self.root, self.planned, {}, 0, 1)
                        with self.assertRaises(writer.InjectedCrash):
                            writer.checkpoint(self.root, plan, [record], point)
                        records, _ = writer.resume(self.root, plan, {}, checkpoint_now=True)
                    self.assertEqual(len(records), 1)
                    self.assertEqual(json.loads((self.root / 'manifest.json').read_text())['journal_sequence'], 1)
                    self.assertTrue(writer.validate_full(self.root, plan, {}, reproduce=False)['valid'])

    def test_lock(self):
        with writer.writer_lock(self.root):
            with self.assertRaises(RuntimeError):
                with writer.writer_lock(self.root):
                    pass
            result = subprocess.run([sys.executable, '-c',
                'from pathlib import Path\nfrom schemagan3d.v3_24k_writer import writer_lock\n'
                f'with writer_lock(Path({str(self.root)!r})):\n    pass\n'],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('active writer lock', result.stderr)

    def test_production_gate_requires_separate_approval(self):
        with patch.object(writer, 'validate_plan', lambda *a, **kw: None):
            with self.assertRaises(PermissionError):
                writer.run_approved_production(self.root, {'entries': []},
                                               {'output_root': 'data/synthetic3d_v2/schemagan_v3_24k'})
        self.assertFalse((self.root / '.writer.lock').exists())

    def test_bad_complete_journal_record(self):
        (self.root / 'journal.jsonl').write_text('{"wrong":1}\n')
        with self.assertRaisesRegex(ValueError, 'invalid complete journal'):
            self._recover()
        self.assertTrue((self.root / 'journal.jsonl').exists())

    def test_incomplete_stage_is_preserved_and_not_committed(self):
        stage = self.root / '.staging-interrupted'
        stage.mkdir()
        (stage / 'partial.npy').write_bytes(b'incomplete')
        records, status = self._recover()
        self.assertEqual(records, [])
        self.assertEqual(status['committed'], 0)
        self.assertTrue((stage / 'partial.npy').exists())

    def test_bad_checkpoint_replayed(self):
        plan = {'entries': [self.planned], 'config_sha256': 'fixture'}
        with patch.object(writer, 'make_volume', self._fake):
            record, _ = writer.write_volume(self.root, self.planned, {}, 0, 1)
            writer.checkpoint(self.root, plan, [record])
            (self.root / 'manifest.json').write_text('{bad')
            records, status = writer.resume(self.root, plan, {}, checkpoint_now=True)
        self.assertEqual(len(records), 1)
        self.assertFalse(status['checkpoint_valid'])
        self.assertEqual(len(list(self.root.glob('manifest.invalid-*.json'))), 1)

    def test_checkpoint_journal_mismatch_replayed(self):
        plan = {'entries': [self.planned], 'config_sha256': 'fixture'}
        with patch.object(writer, 'make_volume', self._fake):
            record, _ = writer.write_volume(self.root, self.planned, {}, 0, 1)
        writer.checkpoint(self.root, plan, [record])
        path = self.root / 'manifest.json'
        bad = json.loads(path.read_text())
        bad['journal_last_sha256'] = 'wrong'
        path.write_text(json.dumps(bad))
        records, status = writer.resume(self.root, plan, {}, checkpoint_now=True)
        self.assertFalse(status['checkpoint_valid'])
        self.assertEqual(len(records), 1)
        self.assertTrue(writer.validate_full(self.root, plan, {}, reproduce=False)['valid'])

    def test_content_corruption_detected_without_deleting_evidence(self):
        plan = {'entries': [self.planned], 'config_sha256': 'fixture'}
        for kind in ('array', 'metadata', 'profiles', 'sidecar', 'sidecar_missing', 'missing_file', 'fingerprint'):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    self.root = Path(tmp)
                    with patch.object(writer, 'make_volume', self._fake):
                        record, _ = writer.write_volume(self.root, self.planned, {}, 0, 1)
                    writer.checkpoint(self.root, plan, [record])
                    entry = record['entry']
                    if kind in ('array', 'metadata', 'profiles'):
                        rel = entry['arrays']['Ic'] if kind == 'array' else entry[kind]
                        path = self.root / rel
                        with path.open('ab') as stream:
                            stream.write(b'corrupt')
                    elif kind == 'sidecar':
                        (self.root / entry['realization_id'] / '_entry.json').write_text('{"wrong":1}')
                    elif kind == 'sidecar_missing':
                        (self.root / entry['realization_id'] / '_entry.json').unlink()
                    elif kind == 'missing_file':
                        (self.root / entry['arrays']['Ic']).unlink()
                    else:
                        bad = dict(entry, fingerprint='bad-fingerprint')
                        sidecar = self.root / entry['realization_id'] / '_entry.json'
                        sidecar.write_text(json.dumps(bad))
                        body = {k: v for k, v in record.items() if k != 'record_sha256'}
                        body.update(entry=bad, fingerprint='bad-fingerprint')
                        modified = {**body, 'record_sha256': writer.digest(body)}
                        (self.root / 'journal.jsonl').write_text(writer.canonical(modified) + '\n')
                        writer.checkpoint(self.root, plan, [modified])
                    with self.assertRaises(ValueError):
                        writer.validate_full(self.root, plan, {}, reproduce=False)
                    self.assertTrue((self.root / 'journal.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
