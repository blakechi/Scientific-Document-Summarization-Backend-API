import os
import pytest

from nlps import Article, NLProcessor
from .context import TITLE, ABSTRACT, SECTIONS


def pytest_configure():
    config_path = os.path.join(os.getcwd(), "model_config.yaml")
    nlp = NLProcessor.load_from_config_path(config_path)

    sections = [
        nlp.analyze_corpus(section, corpus_name=section_name)
        for section_name, section in SECTIONS.items()
    ]
    
    article = Article()
    article.title = TITLE
    article.abstract = ABSTRACT
    article.sections = sections
    
    pytest.article = article
    pytest.TITLE = TITLE
    pytest.ABSTRACT = ABSTRACT
    pytest.SECTIONS = SECTIONS


def pytest_addoption(parser):
    parser.addoption("--store-output", action="store_true", default=False, help="Whether to store APIs' outputs")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.store_output
    if 'store_output' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("store_output", [option_value])