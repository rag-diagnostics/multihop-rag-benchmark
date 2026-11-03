"""Offline end-to-end fixture: model replies and NLP resources are substitutes, not observations."""
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rag_utils.io import atomic_json, read_json, JsonlIndex
from rag_utils.tables import read_records
from data_generation.run_pipeline import build
from data_generation.narrative_generation.generate_narratives import run as narrate
from data_generation.qa_generation.generate_qa import run as qa
from data_generation.entity_remapping import remap_entities as remap
from data_generation.multihop_construction import multi_create as multi
from data_generation.dataset_validation import validate_gold as vg
from data_generation.distractor_selection import select_random_distractors as selection
from evaluation.run_dataset import run_dataset
from evaluation.scoring.judge_answers import judge_file
from evaluation.scoring.aggregate import aggregate_file
from tests.fixtures import chain
from tests.test_corrected_evaluation import config


def test_messages_to_constructed_chains_to_evaluation(tmp_path):
    expected=[chain(0),chain(1)]
    documents=[d for c in expected for d in c['hops']['n4']['all_golden_docs']]
    documents += [f'Noise{k} concerned an unrelated harbour contract.' for k in range(10)]
    sources=[{'Serial':f'source_{i:03d}','source':'synthetic_test_only','text':doc} for i,doc in enumerate(documents)]
    inp=tmp_path/'selected.json';atomic_json(inp,sources)
    def narrative_call(model,prompt,**kwargs):
        return next(d for d in documents if prompt.endswith(d+'\n') or ('\n'+d) in prompt)
    def narrate_fixture(rows,out,model):return narrate(rows,out,model=model,call=narrative_call)
    class Tok:
        def __init__(self,text,ws):
            self.text=text;self.whitespace_=ws;self.is_title=False;self.like_num=False;self.pos_='';self.lemma_=text
    def init(seed):
        remap.random.seed(seed);remap.master_entity_map={}
        remap.nlp=lambda text:[Tok(m.group(1),m.group(2)) for m in re.finditer(r'(\S+)(\s*)',text)]
    def remap_fixture(ip,op,seed):
        # The original transform_text still runs; only unavailable NLP assets are substituted.
        from rag_utils.tables import write_records
        result=remap.process_records(read_records(ip),seed=seed,initialize_fn=init)
        write_records(op,result)
        return result
    def qa_call(model,prompt,**kwargs):
        doc=next(d for d in documents if f'**Summary:** "{d}"' in prompt)
        for c in expected:
            if doc==c['hops']['n1']['golden_docs'][0]:
                return f'QUESTION: {c["original_question"]}\nANSWER: {c["answer"]}'
        return f'QUESTION: Which label starts record {documents.index(doc)}?\nANSWER: {doc.split()[0]}'
    def qa_fixture(ip,op,model):return qa(ip,op,model=model,call=qa_call)
    def gold_response(prompt,key):
        q=prompt.split('QUESTION:\n')[-1].strip()
        found=next((c for c in expected if c['original_question']==q),None)
        return json.dumps({'verdict':'PASS' if found else 'FAIL', 'matched_answer':found['answer'] if found else '',
                           'other_answers':[], 'justification':'Deterministic fixture, not an LLM judgment'})
    class ModelFixture:
        def generate_content(self,prompt,**kwargs):
            if 'Extract up to' in prompt:
                q=prompt.split('Question: "',1)[1].split('"',1)[0]
                value=[]
                for c in expected:
                    for depth in range(1,4):
                        if q==c['hops'][f'n{depth}']['question']:
                            value=[c['decomposition'][depth-1]['entity_to_resolve']]
            elif '**Input Data:**' in prompt:
                q=prompt.split('- Current Question: "',1)[1].split('"',1)[0]
                value=None
                for c in expected:
                    for depth in range(1,4):
                        if q==c['hops'][f'n{depth}']['question']:
                            newq=c['hops'][f'n{depth+1}']['question']
                            entity=c['decomposition'][depth-1]['entity_to_resolve']
                            # A valid description that occurs verbatim in the replacement question.
                            desc=newq[len('Which envoy served '):-1]
                            value={'description':desc,'reasoning_step':f'Resolve {entity} from the supplied fixture.', 'new_question':newq}
                assert value
            elif 'Determine if the exact answer' in prompt:
                q=prompt.split('Question: "',1)[1].split('"',1)[0]
                value={'can_answer':False}
                for c in expected:
                    for depth in range(2,5):
                        if q==c['hops'][f'n{depth}']['question'] and all(d in prompt for d in c['hops'][f'n{depth}']['all_golden_docs']):
                            value={'can_answer':True,'answer':c['answer']}
            else:raise AssertionError('Unexpected prompt')
            text=json.dumps(value)
            return SimpleNamespace(text=text,parts=[SimpleNamespace(text=text)])
    async def no_delay(_):pass
    with patch.object(selection,'call_gemini',return_value='UNKNOWN'), patch.object(vg,'API_KEYS',['fixture-not-a-real-key']), patch.object(vg,'call_gemini_rest_api',gold_response), patch.object(asyncio,'sleep',no_delay):
        summary=build(inp,tmp_path/'build', hooks={'narrate':narrate_fixture,'remap':remap_fixture,'qa':qa_fixture,'multihop_model_object':ModelFixture()})
    assert summary['complete_chains_before_pks']==2 and summary['pks_applied'] is False
    constructed=tmp_path/'build/multihop_validated.json'
    assert not (tmp_path/'build/benchmark.json').exists()
    rows=read_json(constructed)
    assert {c['answer'] for c in rows}=={'Arlen0','Arlen1'}
    assert all(len(c['decomposition'])==3 for c in rows)
    # Exercise PKS classification math separately with declared fixture scores (no reference model).
    from parametric_filtering.pks_core import calibration_stats, classify_pks
    stats={'common':calibration_stats([99,101]),'plausible':calibration_stats([49,51]),'nonsense':calibration_stats([-1,1])}
    assert classify_pks(50,stats)[0]=='plausible'
    # Run the actual direct-JSON interface and original judge routing.
    def generator(cfg,prompt,max_tokens):
        i='0' if 'Doran0' in prompt or 'Alderia0' in prompt else '1'
        # Detect the actual question, not another chain in a distractor.
        q=prompt.split('Question: ',1)[1].split('\n',1)[0]
        i='0' if any(name+'0' in q for name in ('Alderia','Belvar','Corven','Doran')) else '1'
        return {'generated_text':f'The envoy was Arlen{i}.','finish_reason':'stop'}
    run=tmp_path/'evaluation';cfg=config(tmp_path)
    result=run_dataset(constructed,cfg,run,family='random',length_policy='none',expected_chains=2,gen=generator)
    assert result['calls_completed']==24
    def judge(q,gold,pred,model):
        return {'normalized_ans':gold,'is_correct':gold in pred,'verdict':'CORRECT' if gold in pred else 'INCORRECT'}
    judged=judge_file(run/'generations.jsonl',run/'judged.jsonl',judge=judge,judge_model='fixture-not-real',policy='legacy_short')
    assert judged['pending_judgments']==24 and judged['judge_errors']==0
    aggregate_file(run/'judged.jsonl',run/'summary')
    assert all(r['is_correct'] is True for r in JsonlIndex(run/'judged.jsonl').rows())
