"""Publishable results with raw responses and reproducible request hashes."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insurance_eval.core import digest, write_json
from insurance_eval.prompts import build_prompt


def export_results(source, destination, source_commit, workflow_url):
    source, destination = Path(source), Path(destination)
    config = json.loads((source/'config.json').read_text())
    backend = json.loads((source/'backend.json').read_text())
    analysis = json.loads((source/'analysis.json').read_text())
    if config['backend']=='fixture' or analysis['kind']!='model_inference':
        raise ValueError('Only real model results can be exported here')
    if destination.exists():
        raise ValueError('Results destination must be new')
    inputs = json.loads((source/'inputs.json').read_text())
    cases = {c['id']: c for c in inputs['cases']}
    records = [json.loads(p.read_text()) for p in sorted((source/'records').glob('*.json'))]
    by_key = {(r['case_id'],r['repetition'],r['method']): r for r in records}
    for r in records:
        if r['kind']!='model_inference':
            raise ValueError('Fixture output mixed into model run')
        draft = None
        if r['method']!='direct':
            draft = by_key[(r['case_id'],r['repetition'],'direct')]['response']['text']
        case = cases[r['case_id']]
        expected = build_prompt(case,inputs['policies'][case['policy_id']],r['method'],draft)
        if expected!=r['messages'] or digest(expected)!=r['request_sha256']:
            raise ValueError('Cannot reconstruct the exact request')
    destination.mkdir(parents=True)
    public_config = {k:v for k,v in config.items() if k!='model_path'}
    public_config['model_path_note'] = 'Machine-local model path omitted; model identity and file hashes retained.'
    write_json(destination/'provenance.json', {'source_commit':source_commit,
        'workflow_url':workflow_url, 'config':public_config, 'backend':backend,
        'saved_records':len(records), 'request_reconstruction':'All saved request hashes verified',
        'response_format':'Original records with redundant messages omitted. Reconstruct prompts from the frozen source commit, data snapshots, method, and shared direct response.'})
    write_json(destination/'analysis.json',analysis)
    rows=json.loads((source/'scored_cases.json').read_text())
    write_json(destination/'scored_cases.json',rows)
    with (destination/'responses.jsonl').open('w') as stream:
        for r in records:
            stream.write(json.dumps({k:v for k,v in r.items() if k!='messages'},ensure_ascii=False)+'\n')
    with (destination/'case_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (destination/'automated-summary.md').write_text((source/'results.md').read_text())
    return {'records_exported':len(records),'destination':str(destination)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--workflow-url',required=True)
    args=parser.parse_args()
    print(json.dumps(export_results(args.source,args.out,args.source_commit,args.workflow_url),indent=2))
