"""Build the 12 AI-drafted development examples, never a held-out test set."""
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insurance_eval.core import digest, write_json


def main():
    rows, keys = [], []
    families = [
        {'family': 'F01', 'policy': 'CPAP',
         'question': 'Is the additional symptom or comorbidity requirement in NCD 240.4 section B.5.b established?',
         'scope': 'Evaluate only the additional symptom or comorbidity condition for AHI 5 through 14. Do not determine overall eligibility, CED eligibility, or payment.',
         'facts': ['The fictional adult has an AHI of 10 events per hour.'],
         'variants': [
             ('supported', 'The clinical record documents hypertension.', [], 'Documented hypertension satisfies an alternative in the named condition.'),
             ('not_supported', 'The clinical record explicitly rules out excessive daytime sleepiness, impaired cognition, mood disorders, insomnia, hypertension, ischemic heart disease, and any history of stroke.', [], 'All alternatives for this named additional condition are explicitly ruled out.'),
             ('insufficient_information', None, ['qualifying_symptom_or_comorbidity'], 'No supplied fact establishes or rules out the listed alternatives.')],
         'refs': ['CPAP-L116'], 'criteria': ['A listed qualifying symptom or comorbidity is documented'],
         'clarification': 'Does the record document any of the symptoms or comorbidities listed in B.5.b?'},
        {'family': 'F02', 'policy': 'CPAP',
         'question': 'Does the sleep test satisfy both physician-order and supervision requirements in section B.4?',
         'scope': 'Evaluate only section B.4. Other coverage requirements and alternative research pathways are outside this question.',
         'facts': ['The sleep test was furnished under appropriate physician supervision.'],
         'variants': [
             ('supported', 'The beneficiary\'s treating physician ordered the sleep test before it was performed.', [], 'Both the prior treating-physician order and supervision are documented.'),
             ('not_supported', 'The record explicitly states that the treating physician did not order the sleep test before it was performed.', [], 'Supervision does not replace the separate prior-order requirement.'),
             ('insufficient_information', None, ['prior_treating_physician_order'], 'Supervision is known, but the required prior order is unknown.')],
         'refs': ['CPAP-L113'], 'criteria': ['Prior order by treating physician', 'Appropriate physician supervision'],
         'clarification': 'Did the treating physician order the sleep test before it was performed?'},
        {'family': 'F03', 'policy': 'LDCT',
         'question': 'Is the current-smoking or recent-cessation criterion satisfied?',
         'scope': 'Evaluate only the smoking recency criterion in NCD 210.14. Do not determine overall screening eligibility or payment.',
         'facts': ['The fictional beneficiary is a former smoker and is not currently smoking.'],
         'variants': [
             ('supported', 'The beneficiary quit smoking 10 years before the proposed screening date.', [], 'The documented cessation date is within the policy interval.'),
             ('not_supported', 'The beneficiary quit smoking 16 years before the proposed screening date.', [], 'The beneficiary neither currently smokes nor quit within the policy interval.'),
             ('insufficient_information', None, ['years_since_quitting'], 'Former-smoker status alone does not establish the recency interval.')],
         'refs': ['LDCT-L110'], 'criteria': ['Current smoker OR quit within the last 15 years'],
         'clarification': 'How many years before the proposed screening did the beneficiary quit smoking?'},
        {'family': 'F04', 'policy': 'LDCT',
         'question': 'Are all five items in the Beneficiary Eligibility Criteria list satisfied?',
         'scope': 'Evaluate only the five-item beneficiary eligibility list. Counseling, facility and radiologist requirements, and actual payment are outside this question.',
         'facts': ['The fictional beneficiary is 65 years old.', 'The record states there are no signs or symptoms of lung cancer.',
                   'The smoking history is 30 pack-years.', 'The beneficiary is a current smoker.'],
         'variants': [
             ('supported', 'The beneficiary has received an order for lung cancer screening with LDCT.', [], 'All five named beneficiary criteria have explicit supporting facts.'),
             ('not_supported', 'The record explicitly states that the beneficiary has not received an order for lung cancer screening with LDCT.', [], 'The order requirement fails despite the other four criteria being established.'),
             ('insufficient_information', None, ['ldct_screening_order'], 'The record does not establish whether the required screening order exists.')],
         'refs': ['LDCT-L105', 'LDCT-L107', 'LDCT-L108', 'LDCT-L109', 'LDCT-L110', 'LDCT-L111'],
         'criteria': ['Age range', 'Asymptomatic status', 'Minimum pack-years', 'Smoking recency', 'LDCT screening order'],
         'clarification': 'Has the beneficiary received an order for lung cancer screening with LDCT?'}]
    assembled = []
    for f in families:
        for label, variant, missing, rationale in f['variants']:
            facts = f['facts'] + ([variant] if variant else [])
            assembled.append((f, label, facts, missing, rationale))
    random.Random(20261113).shuffle(assembled)
    for i, (f, label, facts, missing, rationale) in enumerate(assembled, 1):
        cid = f'D{i:03}'
        rows.append({'id': cid, 'family_id': f['family'], 'split': 'development', 'policy_id': f['policy'],
                     'question': f['question'], 'task_scope': f['scope'],
                     'facts': [{'id': f'F{j}', 'text': t} for j, t in enumerate(facts, 1)]})
        keys.append({'case_id': cid, 'label': label, 'policy_refs': f['refs'], 'required_criteria': f['criteria'],
                     'missing_fields': missing, 'acceptable_clarification': [f['clarification']] if missing else [],
                     'rationale': rationale, 'review_status': 'ai_drafted_pending_independent_review'})
    for path, data in [(ROOT/'data/cases/development.jsonl', rows), (ROOT/'data/gold/development.jsonl', keys)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SystemExit('Refusing to overwrite existing benchmark: ' + str(path))
        path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data))
    paths = sorted(p for p in (ROOT/'data').rglob('*') if p.is_file() and p.name != 'manifest.json')
    write_json(ROOT/'data/manifest.json', {'created_date': '2026-11-13', 'dataset_version': 'development-v0.1',
        'cases': 12, 'families': 4, 'held_out_test_cases': 0, 'independent_review': False,
        'source_format': 'Normalized extraction of CMS A-D policy text, not original HTML',
        'files': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p.read_bytes())} for p in paths]})
    print('Created 12 development cases, four families, two policies. Independent review pending.')


if __name__ == '__main__':
    main()
