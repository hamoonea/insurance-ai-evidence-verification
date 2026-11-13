"""Revalidate saved model responses and reproduce scores without model inference."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insurance_eval.analyze import analyze, export_review
from insurance_eval.core import digest, parse_object, read_jsonl, validate_answer, write_json
from insurance_eval.prompts import build_prompt


def numerically_equal(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            numerically_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            numerically_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def replay(results, out):
    results, out = Path(results), Path(out)
    if out.exists():
        raise ValueError('Choose a new output directory')
    provenance = json.loads((results/'provenance.json').read_text())
    config = provenance['config']
    if config['backend'] == 'fixture' or not config['experimental_results']:
        raise ValueError('Expected real model results')
    for name, expected in config['input_fingerprints'].items():
        if digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('Frozen source changed: ' + name + '; use source commit ' + provenance['source_commit'])
    cases = {c['id']: c for c in read_jsonl(ROOT/'data/cases/development.jsonl')}
    policies = {p['id']: p for p in (json.loads(f.read_text()) for f in (ROOT/'data/policies').glob('*.json'))}
    records = read_jsonl(results/'responses.jsonl')
    lookup = {(r['case_id'], r['repetition'], r['method']): r for r in records}
    expected = {(cid, rep, method) for cid in config['case_ids']
                for rep in range(config['repeats']) for method in config['methods']}
    if len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError('Missing, duplicate or unexpected records')
    for r in records:
        if r['kind'] != 'model_inference':
            raise ValueError('Non-model response in published results')
        case = cases[r['case_id']]
        draft = None if r['method'] == 'direct' else lookup[(r['case_id'], r['repetition'], 'direct')]['response']['text']
        messages = build_prompt(case, policies[case['policy_id']], r['method'], draft)
        if digest(messages) != r['request_sha256']:
            raise ValueError('Request hash mismatch')
        r['messages'] = messages
        try:
            value = validate_answer(parse_object(r['response']['text']), policies[case['policy_id']],
                                    case, structured=r['method'] == 'evidence')
        except ValueError as error:
            if r['status'] != 'invalid_output' or str(error) != r['error']:
                raise ValueError('Invalid-output decision did not reproduce') from error
        else:
            if r['status'] != 'ok' or value != r['validated']:
                raise ValueError('Validated output did not reproduce')
    out.mkdir(parents=True)
    write_json(out/'config.json', config)
    write_json(out/'backend.json', provenance['backend'])
    write_json(out/'inputs.json', {'cases': list(cases.values()), 'policies': policies})
    for r in records:
        write_json(out/'records'/f"{r['case_id']}-{r['repetition']}-{r['method']}.json", r)
    report = analyze(ROOT, out)
    if not numerically_equal(report, json.loads((results/'analysis.json').read_text())):
        raise ValueError('Published metrics did not reproduce')
    if json.loads((out/'scored_cases.json').read_text()) != json.loads((results/'scored_cases.json').read_text()):
        raise ValueError('Published case scores did not reproduce exactly')
    export_review(ROOT, out)
    return {'verified_records': len(records), 'scored_rows': 'exact match',
            'metrics': 'match within 1e-12 across Python versions',
            'new_inference': False, 'review_packet': str(out/'review')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/development-pilot-v1')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.results, args.out), indent=2))
