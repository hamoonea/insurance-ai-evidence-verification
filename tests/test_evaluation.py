import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from insurance_eval.core import (evaluate_logic, parse_object, public_case, read_jsonl,
                                 validate_answer, validate_dataset)
from insurance_eval.prompts import build_prompt
from insurance_eval.run import execute
from insurance_eval.analyze import analyze, metrics, paired_family_bootstrap

ROOT=Path(__file__).resolve().parents[1]


class LogicTests(unittest.TestCase):
    def test_truth_tables(self):
        values=('true','false','unknown')
        expected_all=[['true','false','unknown'],['false','false','false'],['unknown','false','unknown']]
        expected_any=[['true','true','true'],['true','false','unknown'],['true','unknown','unknown']]
        for i,a in enumerate(values):
            for j,b in enumerate(values):
                for op,expected in [('all',expected_all),('any',expected_any)]:
                    with self.subTest(op=op,a=a,b=b):
                        self.assertEqual(evaluate_logic({op:[{'ref':'a'},{'ref':'b'}]}, {'a':a,'b':b}),expected[i][j])

    def test_nested_alternative(self):
        tree={'all':[{'ref':'age'},{'any':[{'ref':'current'},{'ref':'quit'}]}]}
        self.assertEqual(evaluate_logic(tree,{'age':'true','current':'true','quit':'unknown'}),'true')

    def test_reject_empty_or_unknown_logic(self):
        for tree in [{'all':[]},{'eval':'open(file)'},{'ref':'missing'},{'all':[],'any':[]}]:
            with self.subTest(tree=tree),self.assertRaises(ValueError):
                evaluate_logic(tree,{'a':'true'})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.policy={'segments':[{'id':'P1','text':'A documented order is required.'}]}
        self.case={'facts':[{'id':'F1','text':'The order is documented.'}]}
        self.answer={'label':'supported','explanation':'Order documented.','citations':['P1'],
                     'missing_fields':[],'clarification_questions':[],
                     'criteria':[{'id':'C1','condition':'An order is documented','status':'true',
                                  'policy_refs':['P1'],'case_evidence':[{'fact_id':'F1','quote':'The order is documented.'}],
                                  'missing_fields':[]}], 'logic':{'ref':'C1'}}

    def test_valid_evidence(self):
        self.assertEqual(validate_answer(self.answer,self.policy,self.case,True)['label'],'supported')

    def test_fabricated_policy_reference_is_invalid(self):
        self.answer['criteria'][0]['policy_refs']=['P999']
        with self.assertRaises(ValueError):validate_answer(self.answer,self.policy,self.case,True)

    def test_fabricated_fact_reference_is_invalid(self):
        self.answer['criteria'][0]['case_evidence'][0]['fact_id']='F999'
        with self.assertRaises(ValueError):validate_answer(self.answer,self.policy,self.case,True)

    def test_empty_quote_is_invalid(self):
        self.answer['criteria'][0]['case_evidence'][0]['quote']=''
        with self.assertRaises(ValueError):validate_answer(self.answer,self.policy,self.case,True)

    def test_invented_quote_is_invalid(self):
        self.answer['criteria'][0]['case_evidence'][0]['quote']='No order exists.'
        with self.assertRaises(ValueError):validate_answer(self.answer,self.policy,self.case,True)

    def test_no_evidence_downgrades_both_true_and_false(self):
        for value in ('true','false'):
            self.answer['criteria'][0]['status']=value
            self.answer['criteria'][0]['case_evidence']=[]
            r=validate_answer(self.answer,self.policy,self.case,True)
            self.assertEqual(r['label'],'insufficient_information')
            self.assertTrue(r['gate_changes'])
            self.assertNotEqual(r['explanation'],r['raw_explanation'])

    def test_orphan_criterion_cannot_be_ignored(self):
        c=copy.deepcopy(self.answer['criteria'][0]);c['id']='C2';c['status']='unknown'
        self.answer['criteria'].append(c)
        with self.assertRaises(ValueError):validate_answer(self.answer,self.policy,self.case,True)

    def test_duplicate_criteria_rejected(self):
        self.answer['criteria'].append(copy.deepcopy(self.answer['criteria'][0]))
        with self.assertRaises(ValueError):validate_answer(self.answer,self.policy,self.case,True)

    def test_citation_existence_is_not_semantic_proof(self):
        # This intentionally demonstrates the gate's limitation, not correctness:
        # a real quote can still be misinterpreted. Human semantic review is needed.
        self.answer['criteria'][0]['status']='false'
        self.answer['label']='not_supported'
        self.assertEqual(validate_answer(self.answer,self.policy,self.case,True)['label'],'not_supported')


class DataAndPromptTests(unittest.TestCase):
    def test_frozen_dataset_valid(self):
        self.assertEqual(validate_dataset(ROOT)['cases'],12)

    def test_family_label_balance(self):
        cases=read_jsonl(ROOT/'data/cases/development.jsonl')
        gold={g['case_id']:g['label'] for g in read_jsonl(ROOT/'data/gold/development.jsonl')}
        for family in {c['family_id'] for c in cases}:
            labels=[gold[c['id']] for c in cases if c['family_id']==family]
            self.assertCountEqual(labels,['supported','not_supported','insufficient_information'])

    def test_no_gold_or_identifiers_in_prompts(self):
        case=read_jsonl(ROOT/'data/cases/development.jsonl')[0]
        case.update(rationale='SECRET_GOLD_SENTINEL',expected_label='SECRET_LABEL',family_id='SECRET_FAMILY',id='SECRET_ID')
        policy=json.loads((ROOT/'data/policies/cpap.json').read_text())
        for method in ('direct','generic','evidence'):
            text=json.dumps(build_prompt(case,policy,method,'candidate' if method!='direct' else None))
            for secret in ('SECRET_GOLD_SENTINEL','SECRET_LABEL','SECRET_FAMILY','SECRET_ID'):
                self.assertNotIn(secret,text)
        self.assertEqual(set(public_case(case)),{'question','task_scope','facts'})

    def test_strict_json(self):
        self.assertEqual(parse_object('```json\n{"label":"supported"}\n```')['label'],'supported')
        for s in ('[]','{"a":1} {"b":2}','Prefix {"a":1}'):
            with self.assertRaises((ValueError,json.JSONDecodeError)):parse_object(s)

    def test_review_requires_shared_draft(self):
        case=read_jsonl(ROOT/'data/cases/development.jsonl')[0]
        policy=json.loads((ROOT/'data/policies/cpap.json').read_text())
        with self.assertRaises(ValueError):build_prompt(case,policy,'generic')


class MetricTests(unittest.TestCase):
    def test_invalid_outputs_not_counted_as_abstention(self):
        rows=[{'gold':'insufficient_information','prediction':'invalid_output'},
              {'gold':'insufficient_information','prediction':'supported'},
              {'gold':'supported','prediction':'missing_output'}]
        m=metrics(rows)
        self.assertEqual(m['unsupported_definitiveness'],.5)
        self.assertEqual(m['unresolved_case_failure'],1)
        self.assertEqual(m['invalid_or_missing_count'],2)
        self.assertEqual(m['answer_rate_complete_cases'],0)
        self.assertEqual(m['selective_error'],1)

    def test_all_abstaining_has_no_selective_error_denominator(self):
        m=metrics([{'gold':'supported','prediction':'insufficient_information'}])
        self.assertIsNone(m['selective_error'])
        self.assertEqual(m['answer_rate_complete_cases'],0)

    def test_family_bootstrap_deterministic(self):
        rows=[]
        for f in ('F1','F2'):
            for method,pred in [('generic','supported'),('evidence','insufficient_information')]:
                rows.append({'family_id':f,'method':method,'gold':'insufficient_information','prediction':pred})
        a=paired_family_bootstrap(rows,draws=50)
        self.assertEqual(a,paired_family_bootstrap(rows,draws=50))
        self.assertEqual(a['estimate'],-1)
        self.assertEqual(a['families'],2)


class IntegrationTests(unittest.TestCase):
    def test_fixture_resume_shared_draft_and_no_research_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            args=Namespace(root=str(ROOT),out=folder,backend='fixture',model_path=None,
                           max_tokens=2000,temperature=0.0,limit=3,repeats=1,seed=20261113)
            out=execute(args)
            files=list((out/'records').glob('*.json'))
            self.assertEqual(len(files),9)
            before={p.name:p.read_bytes() for p in files}
            execute(args)
            self.assertEqual(before,{p.name:p.read_bytes() for p in files})
            for r in [json.loads(p.read_text()) for p in files]:
                if r['method']!='direct':
                    a=json.loads((out/'records'/f'{r["case_id"]}.0.direct.json').read_text())
                    self.assertIn(a['response']['text'],r['messages'][1]['content'])
            report=analyze(ROOT,out)
            self.assertFalse(report['model_performance_reported'])
            self.assertNotIn('accuracy',json.dumps(report))
            args.max_tokens=100
            with self.assertRaises(ValueError):execute(args)


if __name__=='__main__':
    unittest.main()
