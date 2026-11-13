import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from insurance_eval.run import execute
from insurance_eval.merge import merge_runs
from insurance_eval.analyze import analyze

ROOT=Path(__file__).resolve().parents[1]


class ShardTests(unittest.TestCase):
    def make_shards(self, folder):
        sources=[]
        for family in ('F01','F02','F03','F04'):
            args=Namespace(root=str(ROOT),out=str(Path(folder)/family),backend='fixture',
                model_path=None,max_tokens=2000,temperature=0.0,limit=None,repeats=1,
                seed=20261113,family=family,cpu_threads=4)
            sources.append(execute(args))
        return sources

    def test_disjoint_shards_preserve_all_records_and_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            shards=self.make_shards(folder)
            merged=merge_runs(ROOT,shards,Path(folder)/'merged')
            report=analyze(ROOT,merged)
            self.assertEqual(report['valid_records'],36)
            self.assertFalse(report['model_performance_reported'])
            original={p.name:p.read_bytes() for s in shards for p in (s/'records').glob('*.json')}
            self.assertEqual(original,{p.name:p.read_bytes() for p in (merged/'records').glob('*.json')})

    def test_duplicate_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            shards=self.make_shards(folder)
            with self.assertRaisesRegex(ValueError,'Overlapping'):
                merge_runs(ROOT,shards+[shards[0]],Path(folder)/'bad')

    def test_missing_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            shards=self.make_shards(folder)
            with self.assertRaisesRegex(ValueError,'every development case'):
                merge_runs(ROOT,shards[:-1],Path(folder)/'bad')

    def test_settings_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            shards=self.make_shards(folder)
            p=shards[-1]/'config.json';c=json.loads(p.read_text());c['max_tokens']=1
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'different settings'):
                merge_runs(ROOT,shards,Path(folder)/'bad')

    def test_missing_record_is_preserved_as_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            shards=self.make_shards(folder)
            next((shards[0]/'records').glob('*.json')).unlink()
            merged=merge_runs(ROOT,shards,Path(folder)/'merged')
            status=json.loads((merged/'status.json').read_text())
            self.assertEqual(status['state'],'incomplete')
            self.assertEqual(status['missing_records'],1)
