import json
import pytest

from main import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def write_json(api, result, is_first=False):
    out = {
        api: result
    }
    
    mode = "w" if is_first else "a"
    with open("./test_api_outputs.js", mode) as f:
        json.dump(out, f, indent=4)


@pytest.mark.order(index=1)
def test_process(client, store_output):
    title_ = pytest.TITLE
    abstract_ = pytest.ABSTRACT
    sections_ = pytest.SECTIONS

    response = client.post('/api/process', json=dict(
        title=title_,
        abstract=abstract_,
        sections=sections_,
    ))
    if store_output:
        write_json('/api/process', response.get_json(), is_first=True)

    assert response.status_code == 200


@pytest.mark.order(index=2)
def test_summarize(client, store_output):
    article = pytest.article
    
    # base
    response = client.post('/api/summary/summarize/base', json=dict(
        title=article.title,
    ))
    if store_output:
        write_json('/api/summary/summarize/base', response.get_json())

    assert response.status_code == 200

    # custom
    response = client.post('/api/summary/summarize/custom', json=dict(
        title=article.title,
        sentences=article.sections[1].sentences,
    ))
    if store_output:
        write_json('/api/summary/summarize/custom', response.get_json())

    assert response.status_code == 200

    # update
    response = client.post('/api/summary/summarize/update', json=dict(
        title=article.title,
        sentences=article.sections[1].sentences,
    ))
    if store_output:
        write_json('/api/summary/summarize/update', response.get_json())

    assert response.status_code == 200


def test_reorder(client, store_output):
    article = pytest.article
    
    response = client.get('/api/summary/reorder', json=dict(
        sentences=article.sections[1].sentences,
    ))
    if store_output:
        write_json('/api/summary/reorder', response.get_json())

    assert response.status_code == 200


def test_paraphrase(client, store_output):
    article = pytest.article
    
    response = client.get('/api/summary/paraphrase', json=dict(
        sentences=article.sentences[:10],
        num_candidates=10,
    ))
    if store_output:
        write_json('/api/summary/paraphrase', response.get_json())

    assert response.status_code == 200


def test_concepts(client, store_output):
    article = pytest.article

    response = client.post('/api/concept/concepts/doc', json=dict(
        title=article.title,
        dbpedia_confidence=0.6,
    ))
    if store_output:
        write_json('/api/concept/concepts/doc', response.get_json())

    assert response.status_code == 200

    response = client.post('/api/concept/concepts/summary', json=dict(
        title=article.title,
        dbpedia_confidence=0.6,
    ))
    if store_output:
        write_json('/api/concept/concepts/summary', response.get_json())

    assert response.status_code == 200


def test_concept_sentences(client, store_output):
    article = pytest.article

    response = client.post('/api/concept/sentences', json=dict(
        title=article.title,
        concepts=article.concepts,
    ))
    if store_output:
        write_json('/api/concept/sentences', response.get_json())

    assert response.status_code == 200

    
def test_concept_cooccurance(client, store_output):
    article = pytest.article

    response = client.post('/api/concept/cooccurance', json=dict(
        title=article.title,
    ))
    if store_output:
        write_json('/api/concept/cooccurance', response.get_json())

    assert response.status_code == 200


def test_embeddings(client, store_output):
    article = pytest.article

    # raw
    response = client.post('/api/embedding/embeddings/raw', json=dict(
        texts=["embedding", article.sentences[-1], article.sections[0].text],
    ))
    if store_output:
        write_json('/api/embedding/embeddings/raw', response.get_json())

    assert response.status_code == 200

    # project
    response = client.post('/api/embedding/embeddings/project', json=dict(
        texts=article.concepts,
        projection_dims=2,
        projection_mode="umap"
    ))
    if store_output:
        write_json('/api/embedding/embeddings/project', response.get_json())

    assert response.status_code == 200


def test_similarities(client, store_output):
    article = pytest.article

    response = client.get('/api/embedding/similarities', json=dict(
        sources=article.sentences,
        targets=article.sentences,
    ))
    if store_output:
        write_json('/api/embedding/embeddings/similarities', response.get_json())

    assert response.status_code == 200


def test_focus(client, store_output):
    article = pytest.article

    # test when only one concepts
    response = client.post('/api/embedding/focus', json=dict(
        focus_texts=[article.concepts[0]],
        texts=article.concepts[1:],
        projection_dims=None,  # will use default: 2
        projection_mode="tsne",
    ))
    if store_output:
        write_json('/api/embedding/embeddings/focus', response.get_json())

    assert response.status_code == 200

    # test when more than one concepts
    response = client.post('/api/embedding/focus', json=dict(
        focus_texts=article.concepts[: 3],
        texts=article.concepts[1:],
        projection_dims=None,  # will use default: 2
        projection_mode="tsne",
    ))
    if store_output:
        write_json('/api/embedding/embeddings/focus', response.get_json())

    assert response.status_code == 200


