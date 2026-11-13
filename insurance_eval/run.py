import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from .backends import FixtureBackend, MLXBackend
from .core import digest, parse_object, read_jsonl, validate_answer, validate_dataset, write_json
from .prompts import build_prompt

METHODS = ('direct', 'generic', 'evidence')


def fingerprints(root):
    files = sorted(list((root/'insurance_eval').glob('*.py')) + list((root/'data').rglob('*.json')) +
                   list((root/'data').rglob('*.jsonl')) + list((root/'data/raw').glob('*.txt')))
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in files}


def model_fingerprint(path):
    paths = sorted(p for p in Path(path).iterdir() if p.suffix in ('.json', '.jinja', '.safetensors'))
    if not any(p.suffix == '.safetensors' for p in paths):
        raise RuntimeError('No local model weights found')
    import hashlib
    result = {}
    for p in paths:
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        result[p.name] = h.hexdigest()
    return result


def execute(args):
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    audit = validate_dataset(root)
    cases = read_jsonl(root/'data/cases/development.jsonl')
    if args.limit:
        cases = cases[:args.limit]
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('Positive repeats and limit are required')
    policies = {p['id']: p for p in (json.loads(f.read_text()) for f in (root/'data/policies').glob('*.json'))}
    config = {'schema_version': 1, 'split': 'development', 'backend': args.backend,
              'experimental_results': args.backend == 'mlx', 'case_ids': [c['id'] for c in cases],
              'methods': list(METHODS), 'repeats': args.repeats, 'seed': args.seed,
              'temperature': args.temperature, 'max_tokens': args.max_tokens,
              'model_path': str(Path(args.model_path).resolve()) if args.model_path else None,
              'input_fingerprints': fingerprints(root), 'audit': audit,
              'shared_initial_answer': True, 'gold_used_in_prompts': False,
              'python': sys.version.split()[0], 'platform': platform.platform()}
    out.mkdir(parents=True, exist_ok=True)
    cp = out/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Run configuration or code changed; choose a new output directory')
    write_json(cp, config)
    try:
        if args.backend == 'fixture':
            backend = FixtureBackend()
        else:
            if not args.model_path:
                raise RuntimeError('--model-path must point to existing local MLX weights')
            backend = MLXBackend(args.model_path, args.max_tokens, args.temperature)
            backend.metadata['file_sha256'] = model_fingerprint(args.model_path)
        meta_path = out/'backend.json'
        if meta_path.exists() and json.loads(meta_path.read_text()) != backend.metadata:
            raise RuntimeError('Model/runtime changed; choose a new run directory')
        write_json(meta_path, backend.metadata)
    except Exception as e:
        write_json(out/'status.json', {'state': 'blocked', 'experimental_results': False,
            'reason': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()})
        raise
    # Preserve exact source and prompts without exposing gold to the backend.
    write_json(out/'inputs.json', {'cases': cases, 'policies': policies})
    for repetition in range(args.repeats):
        ordered = list(cases)
        random.Random(args.seed + repetition).shuffle(ordered)
        for case in ordered:
            policy = policies[case['policy_id']]
            draft = None
            reviews = ['generic', 'evidence']
            random.Random(args.seed + repetition + int(case['id'][1:])).shuffle(reviews)
            for method in ['direct'] + reviews:
                path = out/'records'/f'{case["id"]}.{repetition}.{method}.json'
                messages = build_prompt(case, policy, method, draft)
                request_sha = digest(messages)
                if path.exists():
                    record = json.loads(path.read_text())
                    if record['request_sha256'] != request_sha:
                        raise ValueError('Cached request changed')
                else:
                    record = {'case_id': case['id'], 'family_id': case['family_id'],
                              'method': method, 'repetition': repetition, 'kind': backend.kind,
                              'request_sha256': request_sha, 'messages': messages,
                              'timestamp': datetime.now(timezone.utc).isoformat()}
                    try:
                        response = backend.generate(messages, policy=policy, structured=method=='evidence',
                                                    seed=args.seed+repetition)
                        record['response'] = response
                        if response['finish_reason'] != 'stop':
                            raise ValueError('Incomplete generation: ' + str(response['finish_reason']))
                        obj = parse_object(response['text'])
                        record['validated'] = validate_answer(obj, policy, case, structured=method=='evidence')
                        record['status'] = 'ok'
                    except Exception as e:
                        record['status'] = 'invalid_output' if 'response' in record else 'backend_error'
                        record['error'] = str(e)
                    write_json(path, record)
                    print(case['id'], repetition, method, record['status'], flush=True)
                if method == 'direct':
                    # A malformed initial answer still receives both reviews.
                    # A backend failure cannot supply a shared draft.
                    if 'response' not in record:
                        break
                    draft = record['response']['text']
    expected = len(cases) * args.repeats * len(METHODS)
    files = list((out/'records').glob('*.json'))
    states = [json.loads(p.read_text())['status'] for p in files]
    write_json(out/'status.json', {'state': 'complete' if len(files)==expected else 'incomplete',
        'kind': backend.kind, 'expected_records': expected, 'saved_records': len(files),
        'valid_records': states.count('ok'), 'failed_records': len(states)-states.count('ok'),
        'timestamp': datetime.now(timezone.utc).isoformat()})
    return out
