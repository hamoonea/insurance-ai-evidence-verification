"""Evidence checks and three-valued logic. No medical or coverage oracle."""
import hashlib
import json
from pathlib import Path

LABELS = ('supported', 'not_supported', 'insufficient_information')
STATUSES = ('true', 'false', 'unknown')
PUBLIC_KEYS = ('question', 'task_scope', 'facts')


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def public_case(case):
    # Explicit allowlist prevents labels, rationales, family names and case IDs
    # from entering prompts, even when a caller passes a merged record.
    return {k: case[k] for k in PUBLIC_KEYS}


def parse_object(text):
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    elif text.startswith('```\n') and text.endswith('```'):
        text = text[4:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Expected one JSON object')
    return obj


def strings(value, field, nonempty=False):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(field + ' must be a list of nonempty strings')
    if nonempty and not value:
        raise ValueError(field + ' cannot be empty')
    if len(set(value)) != len(value):
        raise ValueError(field + ' contains duplicates')
    return value


def evaluate_logic(node, states, used=None, depth=0):
    """Strong Kleene AND/OR, with no Python eval or model-generated code."""
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        raise ValueError('Malformed or overly deep logic tree')
    op, value = next(iter(node.items()))
    if op == 'ref':
        if not isinstance(value, str) or value not in states:
            raise ValueError('Unknown criterion reference')
        if used is not None:
            used.add(value)
        return states[value]
    if op not in ('all', 'any') or not isinstance(value, list) or not value:
        raise ValueError('Logic requires nonempty all/any arrays or a ref')
    results = [evaluate_logic(x, states, used, depth + 1) for x in value]
    if op == 'all':
        return 'false' if 'false' in results else ('unknown' if 'unknown' in results else 'true')
    return 'true' if 'true' in results else ('unknown' if 'unknown' in results else 'false')


def validate_answer(obj, policy, case, structured=False):
    """Validate syntax/evidence existence, not semantic entailment/completeness."""
    if obj.get('label') not in LABELS:
        raise ValueError('Unknown answer label')
    if not isinstance(obj.get('explanation'), str) or not obj['explanation'].strip():
        raise ValueError('Missing explanation')
    refs = {s['id']: s['text'] for s in policy['segments']}
    facts = {f['id']: f['text'] for f in case['facts']}
    citations = strings(obj.get('citations'), 'citations', nonempty=True)
    if not set(citations).issubset(refs):
        raise ValueError('Unknown policy citation')
    strings(obj.get('missing_fields'), 'missing_fields')
    strings(obj.get('clarification_questions'), 'clarification_questions')
    result = {'raw_label': obj['label'], 'label': obj['label'], 'gate_changes': [],
              'explanation': obj['explanation'], 'citations': citations,
              'missing_fields': obj['missing_fields'],
              'clarification_questions': obj['clarification_questions']}
    if not structured:
        return result
    criteria = obj.get('criteria')
    if not isinstance(criteria, list) or not criteria or len(criteria) > 30:
        raise ValueError('Expected 1 to 30 criteria')
    states, cleaned = {}, []
    for c in criteria:
        if not isinstance(c, dict):
            raise ValueError('Criterion must be an object')
        cid = c.get('id')
        if not isinstance(cid, str) or not cid or cid in states:
            raise ValueError('Missing or duplicate criterion ID')
        if c.get('status') not in STATUSES or not isinstance(c.get('condition'), str) or not c['condition'].strip():
            raise ValueError('Invalid criterion condition/status')
        pref = strings(c.get('policy_refs'), 'policy_refs', nonempty=True)
        if not set(pref).issubset(refs):
            raise ValueError('Invented criterion policy reference')
        evidence = c.get('case_evidence')
        if not isinstance(evidence, list):
            raise ValueError('case_evidence must be a list')
        for e in evidence:
            if not isinstance(e, dict) or e.get('fact_id') not in facts:
                raise ValueError('Invented fact reference')
            q = e.get('quote')
            if not isinstance(q, str) or not q.strip() or q not in facts[e['fact_id']]:
                raise ValueError('Case quote is not an exact nonempty substring')
        missing = strings(c.get('missing_fields'), 'criterion missing_fields')
        status = c['status']
        if status != 'unknown' and not evidence:
            status = 'unknown'
            result['gate_changes'].append(cid + ': definitive status lacks case evidence')
        states[cid] = status
        cleaned.append({**c, 'validated_status': status, 'missing_fields': missing})
    used = set()
    state = evaluate_logic(obj.get('logic'), states, used)
    if used != set(states):
        raise ValueError('Every extracted criterion must appear in logic')
    label = {'true': 'supported', 'false': 'not_supported', 'unknown': 'insufficient_information'}[state]
    if label != obj['label']:
        result['gate_changes'].append('Final label changed by validated three-valued logic')
    result.update(label=label, criteria=cleaned, logic=obj['logic'])
    # Preserve the free-text model explanation as raw evidence; do not present
    # it as a validated explanation after a gate changes the conclusion.
    result['raw_explanation'] = result.pop('explanation')
    result['explanation'] = 'Named policy criteria: ' + label + '. ' + ' '.join(
        c['id'] + '=' + c['validated_status'] + '.' for c in cleaned)
    return result


def validate_dataset(root):
    root = Path(root)
    cases, gold = read_jsonl(root/'data/cases/development.jsonl'), read_jsonl(root/'data/gold/development.jsonl')
    policies = {p['id']: p for p in (json.loads(x.read_text()) for x in sorted((root/'data/policies').glob('*.json')))}
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)) or set(ids) != {g['case_id'] for g in gold} or len(gold) != len(cases):
        raise ValueError('Case and gold IDs are not one-to-one')
    family_splits = {}
    for c in cases:
        if c['policy_id'] not in policies or c['split'] != 'development':
            raise ValueError('Unknown policy or unexpected split')
        if not c['question'] or not c['task_scope'] or not c['facts']:
            raise ValueError('Incomplete case')
        fids = [f['id'] for f in c['facts']]
        if len(set(fids)) != len(fids) or any(not f['text'] for f in c['facts']):
            raise ValueError('Invalid facts')
        family_splits.setdefault(c['family_id'], set()).add(c['split'])
        if any(k in c for k in ('label', 'expected_label', 'rationale', 'missing_fields', 'gold')):
            raise ValueError('Gold field in public case')
    by_id = {c['id']: c for c in cases}
    for g in gold:
        if g['label'] not in LABELS or not g['rationale']:
            raise ValueError('Invalid gold label')
        p = policies[by_id[g['case_id']]['policy_id']]
        if not set(g['policy_refs']).issubset({s['id'] for s in p['segments']}):
            raise ValueError('Invalid gold evidence')
        if g['review_status'] != 'ai_drafted_pending_independent_review':
            raise ValueError('Unexpected review claim')
    if any(len(s) != 1 for s in family_splits.values()):
        raise ValueError('Family crosses splits')
    manifest = json.loads((root/'data/manifest.json').read_text())
    for item in manifest['files']:
        if digest((root/item['path']).read_bytes()) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['path'])
    return {'cases': len(cases), 'families': len(family_splits), 'policies': len(policies),
            'split': 'development', 'independent_review': False}
