"""Score labels mechanically, leave evidence/clarification semantics for reviewers."""
import csv
import json
import random
from collections import Counter
from pathlib import Path
from .core import LABELS, digest, read_jsonl, write_json
from .run import METHODS


def rate(n, d):
    return n/d if d else None


def metrics(rows):
    unknown = [r for r in rows if r['gold']=='insufficient_information']
    complete = [r for r in rows if r['gold']!='insufficient_information']
    definitive = [r for r in rows if r['prediction'] in LABELS[:2]]
    matrix = {g: {p: 0 for p in (*LABELS, 'invalid_output', 'missing_output')} for g in LABELS}
    for r in rows:
        matrix[r['gold']][r['prediction']] += 1
    recalls, f1s = {}, []
    for g in LABELS:
        tp = matrix[g][g]
        fn = sum(matrix[g].values())-tp
        fp = sum(matrix[o][g] for o in LABELS if o!=g)
        recalls[g] = rate(tp, tp+fn)
        f1s.append(rate(2*tp, 2*tp+fp+fn) or 0)
    return {'n': len(rows), 'correct': sum(r['prediction']==r['gold'] for r in rows),
        'accuracy': rate(sum(r['prediction']==r['gold'] for r in rows),len(rows)),
        'unsupported_definitiveness': rate(sum(r['prediction'] in LABELS[:2] for r in unknown),len(unknown)),
        'unresolved_case_failure': rate(sum(r['prediction']!='insufficient_information' for r in unknown),len(unknown)),
        'answer_rate_complete_cases': rate(sum(r['prediction'] in LABELS[:2] for r in complete),len(complete)),
        'selective_error': rate(sum(r['prediction']!=r['gold'] for r in definitive),len(definitive)),
        'unsupported_negative_count': sum(r['prediction']=='not_supported' and r['gold']!='not_supported' for r in rows),
        'invalid_or_missing_count': sum(r['prediction'] not in LABELS for r in rows),
        'macro_f1': sum(f1s)/len(f1s), 'per_class_recall': recalls, 'confusion_matrix': matrix}


def paired_family_bootstrap(rows, seed=20261113, draws=2000):
    """Evidence minus generic unsupported definitiveness; resample whole families."""
    families = sorted({r['family_id'] for r in rows})
    if len(families)<2:
        return {'available': False, 'reason': 'Fewer than two families'}
    grouped = {f: [r for r in rows if r['family_id']==f] for f in families}
    def difference(sample):
        values = []
        for method in ('evidence','generic'):
            u = [r for r in sample if r['method']==method and r['gold']=='insufficient_information']
            if not u: return None
            values.append(sum(r['prediction'] in LABELS[:2] for r in u)/len(u))
        return values[0]-values[1]
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        sample = [r for f in rng.choices(families,k=len(families)) for r in grouped[f]]
        d = difference(sample)
        if d is not None: samples.append(d)
    samples.sort()
    if not samples: return {'available': False, 'reason':'No paired unknown cases'}
    return {'available': True, 'estimate': difference(rows),
            'percentile_95_interval': [samples[int(.025*(len(samples)-1))], samples[int(.975*(len(samples)-1))]],
            'families':len(families),'draws':draws,'seed':seed,
            'interpretation':'Exploratory family bootstrap; correlated variants/repeats preserved; small-sample uncertainty is substantial.'}


def analyze(root, out):
    root, out = Path(root), Path(out)
    config = json.loads((out/'config.json').read_text())
    records = [json.loads(f.read_text()) for f in sorted((out/'records').glob('*.json'))]
    expected = len(config['case_ids'])*config['repeats']*len(METHODS)
    if config['backend']=='fixture':
        report = {'kind':'software_fixture', 'model_performance_reported':False,
                  'expected_records':expected,'saved_records':len(records),
                  'valid_records':sum(r['status']=='ok' for r in records),
                  'note':'This checks pipeline operation only. Fixture labels are not model findings.'}
        write_json(out/'analysis.json',report)
        (out/'results.md').write_text('# Software smoke check\n\nNo model inference was run.\n\n'+
             f'{len(records)}/{expected} software fixture outputs saved; '+
             f'{report["valid_records"]} passed output validation. No performance metrics are reported.\n')
        return report
    if not records:
        raise ValueError('No model outputs; no performance report will be created')
    gold_path = root/'data/gold/development.jsonl'
    if digest(gold_path.read_bytes()) != config['input_fingerprints']['data/gold/development.jsonl']:
        raise ValueError('Gold changed after run configuration was frozen')
    gold = {g['case_id']:g for g in read_jsonl(gold_path)}
    cases = {c['id']:c for c in json.loads((out/'inputs.json').read_text())['cases']}
    lookup = {(r['case_id'],r['repetition'],r['method']):r for r in records}
    if len(lookup)!=len(records): raise ValueError('Duplicate outputs')
    rows=[]
    for cid in config['case_ids']:
        for rep in range(config['repeats']):
            for method in METHODS:
                r=lookup.get((cid,rep,method))
                prediction = 'missing_output' if r is None else ('invalid_output' if r['status']!='ok' else r['validated']['label'])
                rows.append({'case_id':cid,'family_id':cases[cid]['family_id'],'method':method,'repetition':rep,
                             'gold':gold[cid]['label'],'prediction':prediction,
                             'raw_prediction':r.get('validated',{}).get('raw_label','invalid_output') if r else 'missing_output'})
    summaries={m:metrics([r for r in rows if r['method']==m]) for m in METHODS}
    report={'kind':'model_inference','split':'development','independent_review':False,
            'methods':summaries,'paired_comparison':paired_family_bootstrap(rows),
            'raw_evidence_labels':metrics([{**r,'prediction':r['raw_prediction']} for r in rows if r['method']=='evidence']),
            'semantic_evidence_and_clarification_scores':'pending human review',
            'scope':'AI-drafted development set; exploratory label results only, no held-out or expert-validated claims.'}
    write_json(out/'analysis.json',report)
    write_json(out/'scored_cases.json',rows)
    lines=['# Exploratory development results','',report['scope'],'',
           '| Method | Correct | Invalid or missing | Unsupported definitiveness |', '| --- | --- | --- | --- |']
    for method,s in summaries.items():
        v=s['unsupported_definitiveness']
        lines.append(f'| {method} | {s["correct"]}/{s["n"]} | {s["invalid_or_missing_count"]} | '+('N/A' if v is None else f'{v:.1%}')+' |')
    lines += ['', 'See analysis.json for answer rates, selective errors, confusion matrices, and family-level uncertainty.',
              'Human review of evidence meaning, omissions, and clarification questions remains pending.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    return report


def export_review(root, out):
    root,out=Path(root),Path(out)
    records=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
    random.Random(99173).shuffle(records)
    cases={c['id']:c for c in read_jsonl(root/'data/cases/development.jsonl')}
    packets,key=[],[]
    for i,r in enumerate(records,1):
        rid=f'R{i:04}'
        packets.append({'review_id':rid,'case':cases[r['case_id']],
                        'output':r.get('response',{}).get('text','NO OUTPUT'),
                        'final_label':r.get('validated',{}).get('label','invalid_output'),
                        'kind':r['kind']})
        key.append({'review_id':rid,'case_id':r['case_id'],'method':r['method'],'repetition':r['repetition']})
    review=out/'review';review.mkdir(exist_ok=True)
    write_json(review/'blinded_outputs.json',packets)
    write_json(review/'private_method_key.json',key)
    path=review/'annotations.csv'
    if not path.exists():
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['review_id','label_correct','policy_support','case_support','condition_complete',
                             'missing_fields_correct','clarification_useful','notes'])
            writer.writerows([[p['review_id']]+['']*7 for p in packets])
    return {'review_outputs':len(packets),'note':'Method names concealed; output structure may reveal the method. Do not share private_method_key.json with reviewers.'}
