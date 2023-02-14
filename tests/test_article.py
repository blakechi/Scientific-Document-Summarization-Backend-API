import pytest


def test_article_properties():
    article = pytest.article
    
    # sections
    assert article.sections is not None
    assert isinstance(article.sections, list)
    assert len(article.sections) == 6
    assert len(article) == 6

    # sentences
    assert isinstance(article.sentences, list)
    assert article.num_sentences > 0

    # concepts
    assert isinstance(article.concepts, list)
    assert article.num_concepts > 0

    # text
    assert len(article.text) > 0

    # concept_locations
    assert article.concept_locations is not None
    assert isinstance(article.concept_locations, list)
    assert isinstance(article.get_concept_locations([], article.sections), list)
    

def test_article_helper_functions():
    article = pytest.article
    sections_ = pytest.SECTIONS

    assert article.has_section()
    assert article.has_sentence()
    assert article.has_concept()

    #
    assert all(
        True if name == tr_name else False
        for name, tr_name in zip(article.get_section_names(), sections_.keys())
    )
    assert all(
        True if article.get_section_name(idx) == tr_name else False
        for idx, tr_name in enumerate(sections_.keys())
    )

    #
    assert article._get_sentence_idx_offset_by_sections() is not None
    assert isinstance(article._get_sentence_idx_offset_by_sections(), list)
    

def test_concept_to_sentences():
    article = pytest.article

    for idx, concept in enumerate(article.concepts):
        return_idx = True if idx % 2 == 0 else False
        assert article.concept_to_sentences(concept, return_idx) is not None


def test_sentence_to_concepts():
    article = pytest.article

    for idx, sentence in enumerate(article.sentences):
        return_idx = True if idx % 2 == 0 else False
        assert article.sentence_to_concepts(sentence, return_idx) is not None


def test_section_to_sentences():
    article = pytest.article

    for idx, section_name in enumerate(article.get_section_names()):
        return_idx = True if idx % 2 == 0 else False
        assert article.section_to_sentences(section_name, return_idx) is not None


def test_sentence_to_section():
    article = pytest.article

    for idx, sentence in enumerate(article.sentences):
        return_idx = True if idx % 2 == 0 else False
        assert article.sentence_to_section(sentence, return_idx) is not None


def test_section_to_concepts():
    article = pytest.article

    for idx, section_name in enumerate(article.get_section_names()):
        return_idx = True if idx % 2 == 0 else False
        assert article.section_to_concepts(section_name, return_idx) is not None


def test_concept_to_sections():
    article = pytest.article

    for idx, concept in enumerate(article.concepts):
        return_idx = True if idx % 2 == 0 else False
        assert article.concept_to_sections(concept, return_idx) is not None