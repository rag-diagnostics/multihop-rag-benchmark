"""HTTP/client contract checks without network requests, secrets, or model inference."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import pytest
from rag_utils.gemini import generate_text
from evaluation.providers import generate, validate_models, _client_cache
from evaluation.run_inference import make_job, do_job
from evaluation.dataset import tasks_for_chain
from tests.fixtures import chain


def response(finish='STOP',text='Arlen'):
    return SimpleNamespace(ok=True,status_code=200,json=lambda:{'candidates':[{'finishReason':finish,'content':{'parts':[{'text':text}]}}],'usageMetadata':{'promptTokenCount':12,'candidatesTokenCount':2}})


def test_gemini_request_not_gold_or_key_in_url():
    with patch('rag_utils.gemini.requests.post',return_value=response()) as post:
        assert generate_text('fixture-model','answer the question',key='test-only',max_tokens=15)=='Arlen'
        args,kwargs=post.call_args
        assert 'test-only' not in args[0]
        assert kwargs['headers']=={'x-goog-api-key':'test-only'}
        assert kwargs['json']['generationConfig']['maxOutputTokens']==15


def test_gemini_incomplete_construction_rejected_but_generation_recorded():
    with patch('rag_utils.gemini.requests.post',return_value=response('MAX_TOKENS')):
        with pytest.raises(RuntimeError):generate_text('fixture','q',key='test',retries=1)
        result=generate_text('fixture','q',key='test',allow_truncated=True,return_metadata=True,retries=1)
        assert result['finish_reason']=='MAX_TOKENS'


def test_gemini_failure_never_returns_error_string_as_answer():
    resp=SimpleNamespace(ok=False,status_code=401)
    with patch('rag_utils.gemini.requests.post',return_value=resp):
        with pytest.raises(RuntimeError,match='401'):generate_text('fixture','q',key='private-placeholder',retries=1)


def test_api_model_uses_explicit_config(monkeypatch):
    cfg={'provider':'openai_compatible','model_id':'test-id','api_key_env':'TEST_PROVIDER_KEY','base_url':'https://example.invalid/v1','temperature':0.0,'top_p':1.0}
    monkeypatch.setenv('TEST_PROVIDER_KEY','dummy')
    _client_cache().clear()
    c=SimpleNamespace(message=SimpleNamespace(content='Arlen'),finish_reason='stop')
    api=MagicMock()
    with patch.dict(sys.modules, {'openai':SimpleNamespace(OpenAI=api)}):
        api.return_value.chat.completions.create.return_value=SimpleNamespace(choices=[c],usage=None)
        result=generate(cfg,'question without the gold answer',15)
        assert result['generated_text']=='Arlen'
        call=api.return_value.chat.completions.create.call_args.kwargs
        assert call['max_tokens']==15 and call['model']=='test-id'
        assert set(call)=={'model','messages','max_tokens','temperature','top_p'}
    _client_cache().clear()


def test_no_raw_credentials_in_config():
    with pytest.raises(ValueError):validate_models({'models':{'x':{'provider':'google_genai','model_id':'m','api_key':'do-not-commit'}}})


def test_original_regex_command_following():
    row=tasks_for_chain(chain(),0,'irrelevant')[0]
    cfg={'provider':'openai_compatible','model_id':'fixture'}
    job=make_job(row,'fixture',cfg,run_tag='primary',mode='main',max_tokens=15,free_answer=False)
    job['match_policy']='legacy'
    r=do_job(job,gen=lambda *a:{'generated_text':'one-two-three-four'},max_retries=1)
    assert r['command_following'] is False
    assert r['generated_text']==r['generated_answer']=='one-two-three-four'
