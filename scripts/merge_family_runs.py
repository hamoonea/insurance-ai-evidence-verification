"""Combine downloaded family artifacts and run the frozen descriptive analysis."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insurance_eval.merge import merge_runs
from insurance_eval.analyze import analyze, export_review

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New output directory')
    parser.add_argument('shards', nargs='+', help='Directories containing shard config.json files')
    args = parser.parse_args()
    output = merge_runs(ROOT, args.shards, args.out)
    report = analyze(ROOT, output)
    export_review(ROOT, output)
    print(json.dumps(report, indent=2))
