import asyncio
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from rag_utils.tables import ensure_serials, write_records, read_records
from rag_utils.io import atomic_json
from data_generation.multihop_construction import multi_create as multi
from data_generation.multihop_construction.filter_multihop import filter_entries
from data_generation.narrative_generation.generate_narratives import run as narrate
from data_generation.qa_generation.generate_qa import run as qa_run, generate_qa
from data_generation.schema_adapters.prepare_distractor_input import prepare
from data_generation.entity_remapping import remap_entities as remap
from data_generation.llm_diplomacy.game_log_extraction.extract_game_messages import process_game_logs
from tests.fixtures import chain


def test_long_narrative_json_lossless_and_serial(tmp_path):
    rows=[{'game_id':'17','text':'x'*40000}]
    out=tmp_path/'narratives.json'
    result=narrate(rows,out,call=lambda model,prompt,**kw:'y'*40000)
    assert result[0]['Serial']=='sample_000001'
    assert len(read_records(out)[0]['text'])==len(read_records(out)[0]['Summary'])==40000
    with pytest.raises(ValueError,match='32767'):write_records(tmp_path/'bad.xlsx',result)
    assert not (tmp_path/'bad.xlsx').exists()


def test_source_ids_safe_and_unique():
    with pytest.raises(ValueError):ensure_serials([{'Serial':'../secret'}])
    with pytest.raises(ValueError):ensure_serials([{'Serial':'same'},{'Serial':'same'}])


def test_construction_errors_not_saved_as_narratives(tmp_path):
    out=tmp_path/'n.json'
    def broken(*a,**kw):raise RuntimeError('synthetic provider error')
    with pytest.raises(RuntimeError):narrate([{'text':'Some messages'}],out,call=broken)
    assert not out.exists()
    assert json.loads(Path(str(out)+'.status.json').read_text())['completed']==0


def test_qa_adds_id_and_stable_resume(tmp_path):
    inp=tmp_path/'in.json';out=tmp_path/'out.json'
    atomic_json(inp,[{'Summary':'Gabon proposed a pact.'},{'Summary':'Durham signed a treaty.'}])
    calls=[]
    def call(model,prompt,**kw):
        calls.append(prompt)
        a='Gabon' if '**Summary:** "Gabon' in prompt else 'Durham'
        return f'QUESTION: What did {a} do?\nANSWER: {a}'
    qa_run(inp,out,call=call)
    qa_run(inp,out,call=call)
    result=read_records(out)
    assert len(calls)==2 and result[0]['Document']=='Gabon proposed a pact.'
    assert result[0]['question_type']=='What' and result[1]['question_type']=='Where'


def test_qa_rejects_wrong_document_answer():
    with pytest.raises(ValueError):generate_qa('Durham signed a pact.','What',call=lambda *a,**kw:'QUESTION: Which place?\nANSWER: Gabon')


def test_prepare_prefers_transformed_text_and_loader(tmp_path):
    inp=tmp_path/'in.json';out=tmp_path/'qa.json'
    atomic_json(inp,[{'text':'raw France','Summary':'transformed Alderia','question':'Which?','answer':'Alderia'}])
    prepare(inp,out)
    assert multi.load_global_docs(str(out))==['transformed Alderia']
    workbook=tmp_path/'both.xlsx';write_records(workbook,read_records(out))
    assert multi.load_global_docs(str(workbook))==['transformed Alderia']


def test_loader_never_guesses_raw_text(tmp_path):
    p=tmp_path/'bad.json';atomic_json(p,[{'text':'raw only'}])
    with pytest.raises(ValueError):multi.load_global_docs(str(p))


def test_remap_rejects_already_assembled_data():
    with pytest.raises(ValueError):remap.process_records([{'Summary':'France','question':'Which nation?'}],initialize_fn=lambda seed:pytest.fail('should fail before model load'))


def test_remap_document_alignment(tmp_path):
    class Tok:
        def __init__(self,t,w=' '):self.text=t;self.whitespace_=w;self.is_title=t.istitle();self.like_num=t.isdigit();self.pos_='';self.lemma_=t
    def init(seed):
        remap.random.seed(seed);remap.master_entity_map={'France':'Alderia'}
        remap.nlp=lambda text:[Tok(x,'' if i==len(text.split())-1 else ' ') for i,x in enumerate(text.split())]
    result=remap.process_records([{'Summary':'France supported a pact','text':'raw France'}],initialize_fn=init)
    assert result[0]['Summary']==result[0]['Document']=='Alderia supported a pact'
    assert result[0]['text']=='raw France'


def test_multihop_rejects_wrong_answer():
    async def answer(q,docs,gold,model):return (len(docs)>1,'Wrong')
    with patch.object(multi,'can_answer_with_docs',answer):
        ok,why=asyncio.run(multi.verify_russian_doll_hop('Who served France?','Who served the northern nation?',['base'],'link','France','Arlen',None))
    assert not ok and why=='returned_answer_disagrees_with_gold'


def test_multihop_requires_actual_bridge():
    async def answer(*a):pytest.fail('no API calls for invalid structural replacement')
    with patch.object(multi,'can_answer_with_docs',answer):
        ok,why=asyncio.run(multi.verify_russian_doll_hop('Who served France?','Who served the northern nation?',['base'],'link','Sudan','Arlen',None))
    assert not ok and why=='bridge_absent_from_previous_question'


def test_multihop_accepts_necessary_evidence_only():
    async def answer(q,docs,gold,model):return (len(docs)==2,gold if len(docs)==2 else None)
    with patch.object(multi,'can_answer_with_docs',answer):
        assert asyncio.run(multi.verify_russian_doll_hop('Who served France?','Who served the northern nation?',['base'],'link','France','Arlen',None))==(True,'ok')


def test_failed_validation_does_not_pass_as_no_answer():
    async def answer(q,docs,gold,model):
        if len(docs)==2:return True,gold
        raise RuntimeError('test transport failure')
    with patch.object(multi,'can_answer_with_docs',answer):
        ok,why=asyncio.run(multi.verify_russian_doll_hop('Who served France?','Who served the northern nation?',['base'],'link','France','Arlen',None))
    assert not ok and why=='answerability_check_failed'


def test_malformed_bool_rejected():
    async def raw(*a,**kw):return '{"can_answer":"false"}'
    with patch.object(multi,'safe_generate',raw):
        with pytest.raises(ValueError):asyncio.run(multi.can_answer_with_docs('q',['doc'],'gold',None))


def test_filter_rejects_incomplete_records():
    good=chain(family='random');bad={'original_question':'Missing','hops':{'n1':{}}}
    kept,rejected=filter_entries([good,bad]);assert kept==[good] and len(rejected)==1


def test_phase_and_unicode_extraction(tmp_path):
    inp=tmp_path/'in';(inp/'game_000').mkdir(parents=True)
    atomic_json(inp/'game_000/lmvsgame.json',{'phases':[{'name':'W1901A','messages':[{'sender':'FRANCE','recipient':'GERMANY','message':'Caf\u00e9 diplomacy'}]}]})
    out=tmp_path/'out';info=process_game_logs(inp,out)
    text=(out/'game_000_messages.txt').read_text()
    assert 'Winter 1901' in text and 'Caf\u00e9' in text and len(info)==1


def test_bad_phase_not_invented(tmp_path):
    inp=tmp_path/'in';(inp/'game_000').mkdir(parents=True)
    atomic_json(inp/'game_000/lmvsgame.json',{'phases':[{'messages':[{'sender':'FRANCE','recipient':'GERMANY','message':'hello'}]}]})
    with pytest.raises(RuntimeError):process_game_logs(inp,tmp_path/'out')


def test_pks_default_stops_on_failure():
    p=Path(__file__).resolve().parents[1]/'parametric_filtering/pks_filtering/00_para14.py'
    spec=importlib.util.spec_from_file_location('pks_filter_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    stats={cat:{'mean':x,'std':1} for cat,x in [('common',100),('plausible',50),('nonsense',0)]}
    def bad(*a):raise RuntimeError('test scoring failure')
    with pytest.raises(RuntimeError):m.filter_items([{'original_question':'Q','answer':'A'}],stats,bad)
    kept,removed,decisions=m.filter_items([{'original_question':'Q','answer':'A'}],stats,bad,on_error='keep')
    assert len(kept)==1 and decisions['Q']['error']=='RuntimeError'


def test_calibration_checksum_detects_mixed_scores(tmp_path):
    import hashlib
    from parametric_filtering.pks_core import DEFAULT_CALIBRATION_FILENAMES,load_calibration
    hashes={}
    for cat,filename in DEFAULT_CALIBRATION_FILENAMES.items():
        p=tmp_path/filename;atomic_json(p,{'pmi_classic':[1,2,3]})
        hashes[cat]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(tmp_path/'calibration_summary.json',{'score_sha256':hashes,'model':'fixture'})
    assert load_calibration(tmp_path)['common']['n']==3
    atomic_json(tmp_path/DEFAULT_CALIBRATION_FILENAMES['common'],{'pmi_classic':[9,10,11]})
    with pytest.raises(ValueError,match='checksum'):load_calibration(tmp_path)


def test_calibration_publishes_only_complete_bundle(tmp_path):
    path=Path(__file__).resolve().parents[1]/'parametric_filtering/baseline_calibration/para11.py'
    spec=importlib.util.spec_from_file_location('calibration_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    inp=tmp_path/'input';inp.mkdir();out=tmp_path/'out'
    for filename in m.DEFAULT_QA_FILENAMES.values():atomic_json(inp/filename,[{'question':'one','answer':'a'},{'question':'two','answer':'b'}])
    with patch.object(m,'load_model_and_tokenizer',return_value=(None,None)),patch.object(m,'nll_scorer',return_value=None),patch.object(m,'score_qa',side_effect=[1.,2.,RuntimeError('fixture failure')]):
        with pytest.raises(RuntimeError):m.run('fixture',inp,out)
    assert not out.exists()
    with patch.object(m,'load_model_and_tokenizer',return_value=(None,None)),patch.object(m,'nll_scorer',return_value=None),patch.object(m,'score_qa',side_effect=[1.,2.,3.,4.,5.,6.]):
        m.run('fixture',inp,out)
    manifest=json.loads((out/'calibration_summary.json').read_text())
    assert manifest['status']=='complete' and len(manifest['score_sha256'])==3
    with pytest.raises(ValueError,match='not empty'):m.run('fixture',inp,out)
