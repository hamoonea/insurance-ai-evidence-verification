"""Merge disjoint family runs without silently mixing settings or dropping failures."""
import json
from pathlib import Path
from .core import digest, read_jsonl, write_json
from .run import fingerprints, METHODS


def merge_runs(root, sources, destination):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise ValueError('Merge destination must be new')
    shards = []
    ids, records = set(), {}
    baseline = backend = None
    for source in sources:
        source = Path(source)
        config = json.loads((source/'config.json').read_text())
        meta = json.loads((source/'backend.json').read_text())
        inputs = json.loads((source/'inputs.json').read_text())
        common = {k: v for k, v in config.items() if k not in ('case_ids', 'family_filter', 'platform')}
        if baseline is None:
            baseline, backend = common, meta
        if common != baseline or meta != backend:
            raise ValueError('Shards use different settings, model, or runtime')
        if ids.intersection(config['case_ids']):
            raise ValueError('Overlapping case IDs across shards')
        ids.update(config['case_ids'])
        shards.append({'source_name': source.name, 'config': config, 'backend': meta})
        if config['input_fingerprints'] != fingerprints(root):
            raise ValueError('Source/code fingerprints differ from frozen run')
        expected = {(cid, rep, method) for cid in config['case_ids']
                    for rep in range(config['repeats']) for method in METHODS}
        for path in sorted((source/'records').glob('*.json')):
            r = json.loads(path.read_text())
            key = (r['case_id'], r['repetition'], r['method'])
            if key not in expected or key in records or digest(r['messages']) != r['request_sha256']:
                raise ValueError('Unexpected, duplicate, or corrupted request record')
            records[key] = r
        live_cases = {c['id']: c for c in read_jsonl(root/'data/cases/development.jsonl')}
        if inputs['cases'] != [live_cases[cid] for cid in config['case_ids']]:
            raise ValueError('Frozen case snapshot mismatch')
        policies = {p['id']: p for p in (json.loads(f.read_text()) for f in (root/'data/policies').glob('*.json'))}
        if inputs['policies'] != policies:
            raise ValueError('Frozen policy snapshot mismatch')
    cases = read_jsonl(root/'data/cases/development.jsonl')
    if ids != {c['id'] for c in cases}:
        raise ValueError('Shards must cover every development case exactly once')
    config = {**baseline, 'case_ids': [c['id'] for c in cases], 'family_filter': None,
              'platform': 'See merged_shards.json for original runner platforms'}
    write_json(destination/'config.json', config)
    write_json(destination/'backend.json', backend)
    write_json(destination/'merged_shards.json', shards)
    write_json(destination/'inputs.json', {'cases': cases, 'policies': policies})
    for (cid, rep, method), r in records.items():
        write_json(destination/'records'/f'{cid}.{rep}.{method}.json', r)
    expected_count = len(cases)*config['repeats']*len(METHODS)
    write_json(destination/'status.json', {'state': 'complete' if len(records)==expected_count else 'incomplete',
        'kind': 'software_fixture' if config['backend']=='fixture' else 'model_inference',
        'expected_records': expected_count, 'saved_records': len(records),
        'valid_records': sum(r['status']=='ok' for r in records.values()),
        'missing_records': expected_count-len(records)})
    return destination
