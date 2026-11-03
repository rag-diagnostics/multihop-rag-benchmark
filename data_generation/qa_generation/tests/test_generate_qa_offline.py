"""QA transport is mocked; original prompt and checked outputs are tested."""
import json
import pytest
from data_generation.qa_generation.generate_qa import generate_qa, parse_qa, run
from data_generation.qa_generation.prompt import build_prompt
from rag_utils.io import atomic_json


def test_prompt_retained():
    p=build_prompt('Gabon proposed a pact.','Which town')
    assert 'STRICT LIMIT: One to two words maximum' in p
    assert 'Gabon proposed a pact.' in p
    assert 'Your 1-2 word answer here' in p


def test_model_setting_and_output():
    calls=[]
    def fake(model,prompt,**kw):calls.append((model,kw));return 'QUESTION: Which town proposed a pact?\nANSWER: Gabon'
    assert generate_qa('Gabon proposed a pact.','Which town',call=fake)==('Which town proposed a pact?','Gabon')
    assert calls[0][0]=='gemini-2.5-pro'


def test_invalid_span_fails():
    with pytest.raises(ValueError):generate_qa('Durham signed a treaty.','Which town',call=lambda *a,**kw:'QUESTION: Which town?\nANSWER: Gabon')


def test_no_silent_parse_success():
    with pytest.raises(ValueError):parse_qa('The answer is not available.')
