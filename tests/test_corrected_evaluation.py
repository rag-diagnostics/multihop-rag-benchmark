import copy
import json
from pathlib import Path
import pytest
from rag_utils.io import iter_json_array, atomic_json, JsonlIndex
from tools.download_dataset import download, verify
from evaluation.dataset import check_chain, audit_dataset, tasks_for_chain, cap_distractors
from evaluation.run_dataset import run_dataset
from evaluation.scoring.judge_answers import classify, judge_file
from evaluation.scoring.aggregate import aggregate, aggregate_file
from tests.fixtures import chain


@pytest.mark.parametrize("chunk", [1, 2, 7, 101, 65536])
def test_streaming_nested_unicode(tmp_path, chunk):
    data=[{"value":'[a, b] {"quoted": "yes"}',"utf":"\u03b1\u00e9", "nested":{"a":[1,2]}}]*3
    p=tmp_path/'a.json';atomic_json(p,data)
    assert list(iter_json_array(p,chunk))==data


@pytest.mark.parametrize("text", ['<html>denied</html>', '[{"a":1}', '[{},]', '[{}]oops', '[{"a":1,"a":2}]', '[{"a":NaN}]', '{"hops":{}}'])
def test_invalid_json_rejected(tmp_path,text):
    p=tmp_path/'a.json';p.write_text(text)
    with pytest.raises(ValueError):list(iter_json_array(p,4))


def test_download_count_and_atomicity(tmp_path):
    p=tmp_path/'dataset.json';p.write_text('[{"old":true}]')
    def bad(dest):dest.write_text('<html>denied</html>');return str(dest)
    with pytest.raises(ValueError):download(p,expected=2,force=True,fetch=bad)
    assert json.loads(p.read_text())==[{"old":True}]
    def good(dest):atomic_json(dest,[chain(),chain(1)]);return str(dest)
    info=download(p,expected=2,force=True,fetch=good)
    assert info['records']==2 and info['downloaded']
    with pytest.raises(ValueError):verify(p,expected=528)


def test_schema_and_bridge_audit(tmp_path):
    c=chain();assert check_chain(c,'relevant')==[]
    c['decomposition'][0]['entity_to_resolve']='Missingland'
    c['hops']['n2']['bridge_entity_replaced']='Missingland'
    assert any(x['issue']=='bridge_absent_from_previous_question' for x in check_chain(c,'relevant'))
    c['hops']['n3']['all_golden_docs']=[]
    with pytest.raises(ValueError):check_chain(c,'relevant')


def test_both_families_no_silent_mixing():
    a=tasks_for_chain(chain(),0,'relevant');b=tasks_for_chain(chain(),0,'irrelevant')
    assert len(a)==len(b)==12
    assert all(len(r['documents'])==11 for r in a+b)
    assert a[0]['documents']!=b[0]['documents']
    assert a[0]['distractor_family']=='relevant'


def test_raw_text_and_seed_preserved():
    c=chain();before=copy.deepcopy(c)
    a=tasks_for_chain(c,3,'irrelevant',42);b=tasks_for_chain(c,3,'irrelevant',42)
    assert a==b and c==before
    assert a[1]['gold_positions_1_based']==[6]
    assert a[2]['gold_positions_1_based']==[11]


def test_length_cap_explicit():
    c=chain()
    for h in c['hops'].values():
        key='context_relevant_remaining' if 'context_relevant_remaining' in h else 'context_relevant'
        h[key]=[text+' long '*500 for text in h[key]]
    raw,_=cap_distractors(c,'relevant','none')
    capped,info=cap_distractors(c,'relevant','legacy_word_min')
    assert len(capped['hops']['n1']['context_distractors'][0])<len(raw['hops']['n1']['context_distractors'][0])
    assert capped['hops']['n1']['golden_docs']==raw['hops']['n1']['golden_docs']
    assert info['n1']['gold_truncated'] is False


def config(tmp):
    p=tmp/'models.json';atomic_json(p,{"models":{"fixture":{"provider":"openai_compatible","model_id":"fixture-not-real","base_url":"https://example.invalid/v1","api_key_env":"FIXTURE_TEST_ONLY","max_tokens":15}}});return p


def generation(cfg,prompt,max_tokens):
    assert 'golden_answer' not in prompt
    return {"generated_text":"Arlen0","finish_reason":"stop","prompt_tokens":99,"completion_tokens":3}


def test_direct_run_and_resume(tmp_path):
    p=tmp_path/'data.json';atomic_json(p,[chain()]);models=config(tmp_path);out=tmp_path/'run'
    calls=[]
    def gen(*a):calls.append(a);return generation(*a)
    first=run_dataset(p,models,out,family='irrelevant',length_policy='none',gen=gen)
    assert first['calls_completed']==12 and len(calls)==12
    again=run_dataset(p,models,out,family='irrelevant',length_policy='none',gen=gen)
    assert again['resumed']==12 and len(calls)==12
    with pytest.raises(ValueError):run_dataset(p,models,out,family='irrelevant',length_policy='none',max_tokens=16,gen=gen)


def test_preflight_full_count_before_calls(tmp_path):
    p=tmp_path/'data.json';atomic_json(p,[chain()]);calls=[]
    with pytest.raises(ValueError):run_dataset(p,config(tmp_path),tmp_path/'run',family='irrelevant',length_policy='none',expected_chains=528,gen=lambda *x:calls.append(x))
    assert calls==[]


def test_strict_quality_gate_and_explicit_report(tmp_path):
    p=tmp_path/'data.json';c=chain();c['hops']['n2']['bridge_entity_replaced']='Invalid';atomic_json(p,[c]);models=config(tmp_path)
    with pytest.raises(ValueError):run_dataset(p,models,tmp_path/'strict',family='irrelevant',length_policy='none',gen=generation)
    ok=run_dataset(p,models,tmp_path/'report',family='irrelevant',length_policy='none',audit_policy='report',gen=generation)
    assert ok['calls_completed']==12
    assert next(JsonlIndex(tmp_path/'report/generations.jsonl').rows())['dataset_quality_status']=='requires_review'


def result(pred='Fall'):
    return {"request_id":"x","request_fingerprint":"same","status":"ok","mode":"main", "question":"Which season?", "golden_answer":"Autumn", "generated_text":pred,"generated_answer":pred,"match_policy":"legacy"}


def test_explicit_legacy_vs_semantic_policy():
    calls=[]
    def judge(*a):calls.append(a);return {'normalized_ans':'Fall','is_correct':True,'verdict':'CORRECT'}
    assert classify(result(),judge,policy='legacy_short')['is_correct'] is False
    assert calls==[]
    assert classify(result(),judge,policy='semantic_all')['is_correct'] is True
    assert len(calls)==1


def test_short_alias_exact_needs_no_judge():
    r=result();r['golden_answer']='Autumn, Fall'
    assert classify(r,lambda *a:pytest.fail('unneeded judge'),policy='legacy_short')['is_correct'] is True


@pytest.mark.parametrize('obj', [
    {'normalized_ans':'Fall','is_correct':'false','verdict':'CORRECT'},
    {'normalized_ans':'Fall','is_correct':False,'verdict':'CORRECT'},
    {'normalized_ans':'too many tokens in answer','is_correct':True,'verdict':'CORRECT'},
    {'normalized_ans':'HALLUCINATED','is_correct':True,'verdict':'CORRECT'},
    {}])
def test_bad_judge_is_error_not_wrong(obj):
    r=classify(result(),lambda *a:obj,retries=1,policy='semantic_all')
    assert r['judgment_status']=='judge_error' and r['is_correct'] is None


def test_judge_resume_includes_model_policy_and_prediction(tmp_path):
    p=tmp_path/'in.jsonl';out=tmp_path/'out.jsonl';p.write_text(json.dumps(result())+'\n')
    calls=[]
    def judge(*a):calls.append(a);return {'normalized_ans':'Fall','is_correct':True,'verdict':'CORRECT'}
    judge_file(p,out,judge=judge,judge_model='fixture-one',policy='semantic_all')
    judge_file(p,out,judge=judge,judge_model='fixture-one',policy='semantic_all')
    assert len(calls)==1
    judge_file(p,out,judge=judge,judge_model='fixture-two',policy='semantic_all')
    assert len(calls)==2
    p.write_text(json.dumps(result('The answer is Fall'))+'\n')
    judge_file(p,out,judge=judge,judge_model='fixture-two',policy='semantic_all')
    assert len(calls)==3


def test_aggregate_full_pipeline_and_missing_coverage(tmp_path):
    p=tmp_path/'data.json';atomic_json(p,[chain()]);out=tmp_path/'run'
    run_dataset(p,config(tmp_path),out,family='irrelevant',length_policy='none',gen=generation)
    judge_file(out/'generations.jsonl',out/'judged.jsonl',judge=lambda *a:pytest.fail('exact match'),policy='legacy_short')
    report=aggregate_file(out/'judged.jsonl',out/'summary')
    assert report['coverage']['missing']==0
    assert all(r['accuracy']==1 for r in report['by_condition'])
    assert all(r['ccs_n4']==1 for r in report['chain_completion'])
    lines=(out/'judged.jsonl').read_text().splitlines();(out/'judged.jsonl').write_text('\n'.join(lines[:-1])+'\n')
    report=aggregate_file(out/'judged.jsonl',out/'partial')
    assert report['coverage']['missing']==1
    assert all(r['accuracy'] is None for r in report['by_condition'])


def test_error_origin_survives_streaming(tmp_path):
    c=tasks_for_chain(chain(),0,'irrelevant')[0]
    r={**c,'request_id':'x','model_key':'fixture','status':'ok','mode':'main','golden_answer':'Arlen0','is_correct':False,'judge_verdict':'INCORRECT','normalized_ans':'Independent','generated_answer':'Independent','judgment_status':'judge'}
    p=tmp_path/'a.jsonl';p.write_text(json.dumps(r)+'\n')
    report=aggregate_file(p,tmp_path/'summary')
    assert report['by_condition'][0]['n_distracted']==1
