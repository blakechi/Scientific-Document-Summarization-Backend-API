'''main app declaration'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from itertools import combinations
from typing import Dict, List, Literal, Mapping, Optional, Any

import torch
from flask_cors import CORS
from flask import Flask, jsonify, send_from_directory, request

from nlps import NLProcessor, Article

import json

##
## Caches
##
ARTICLES: Dict[str, Article] = {}
API: Dict[str, Dict[str, Any]] = {}
DEFAULT_CONFIG_PATH: str = os.path.join(os.getcwd(), "model_config.yaml")


##
## Helpers
##
def get_article(title: str) -> Article:
    global ARTICLES
    assert title in ARTICLES, (
        f"Article: {title} doesn't exist.",
        "Please call the API: `process` first."
    )

    return ARTICLES[title]


def get_api_cache(api_path: str) -> Dict[str, Any]:
    global API
    if api_path not in API:
        API[api_path] = {}

    return API[api_path]


def update_api_cache(api_path: str, keys: List[str], values: List[Any]) -> None:
    global API
    if api_path not in API:
        API[api_path] = {}
    
    API[api_path].update({key: value for key, value in zip(keys, values)})


def is_list_of_str(target: List[str]) -> bool:
    return  isinstance(target, list) and all(isinstance(ele, str) for ele in target)


def concept_location_transform(lst_concept_per_sec: List[Mapping], concept_tfidfs: Dict[str, Mapping]) -> Mapping:
    # inject concept frequency and tfidf
    for section in lst_concept_per_sec:
        for concept_info in section["concepts"]:
            concept_name = concept_info["concept"]
            if concept_name in concept_tfidfs:
                concept_info["frequency"] = concept_tfidfs[concept_name]["frequency"]
                concept_info["tfidf"] = concept_tfidfs[concept_name]["tfidf"]

    # dictionary of concept locations
    dict_concept_loc = {}
    for i_sec, sec in enumerate(lst_concept_per_sec):
        for i_concept, concept in enumerate(sec["concepts"]):
            concept_name = concept["concept"]
            if concept_name in concept_tfidfs:
                if concept_name in dict_concept_loc:
                    # format of the location - 3d array: [sectionId, conceptId in conecptbySec, [locations]]
                    dict_concept_loc[concept_name].append([i_sec, i_concept, concept["locations"]])
                else:
                    # format of the location - 3d array: [sectionId, conceptId in conecptbySec, [locations]]
                    dict_concept_loc[concept_name] = [[i_sec, i_concept, concept["locations"]]]

    return {
        "lst_concept_per_sec": lst_concept_per_sec,
        "dict_concept_loc": dict_concept_loc,
    }


##
## NLP
##
nlp = NLProcessor.load_from_config_path(DEFAULT_CONFIG_PATH)


##
## APP creation
##
app = Flask(__name__)
# app = Flask(__name__, static_folder='../build')
CORS(app)


##
## API routes
##
@app.route('/api/process', methods=['PUT', 'POST'])
def process():
    data = request.get_json()
    title: str = data["title"]
    abstract: str = data["abstract"]
    _sections: Dict[str, str] = data["sections"]

    assert isinstance(title, str)
    assert isinstance(abstract, str)
    assert isinstance(_sections, dict), (
        f"`sections` should be a dict that section names are keys and section texts are values"
    )

    sections = [
        nlp.analyze_corpus(section, corpus_name=section_name)
        for section_name, section in _sections.items()
    ]

    article = Article()
    article.title = title
    article.abstract = abstract
    article.sections = sections

    # sentence embeddings
    sentence_embeddings = nlp.encode_text(article.sentences).cpu().detach().numpy()
    article.sentence_embeddings = sentence_embeddings
    
    global ARTICLE
    ARTICLES[title] = article

    results = [
        {
            "section": section.name,
            "section_idx": section.idx,
            "sentences": section.sentences
        }
        for section in article.sections
    ]

    return jsonify(results)


@app.route('/api/concept/concepts/<string:mode>', methods=['POST'])
def concepts(mode: Literal["doc", "summary"]):
    assert mode in ["doc", "summary"]

    data = request.get_json()
    title: str = data["title"]
    dbpedia_confidence: float = data["dbpedia_confidence"]

    assert isinstance(title, str)
    assert isinstance(dbpedia_confidence, float)

    article = get_article(title)

    concepts = None
    concept_locations = None
    corpus = None
    if mode == "doc":
        # if dbpedia_confidence != nlp.dbpedia_confidence:
        #     article.sections = [nlp.analyze_corpus(section.text, dbpedia_confidence) for section in article.sections]
        concepts = article.filter_concepts_by_similarity(dbpedia_confidence)
        concept_locations = article.concept_locations
        corpus = [section.text for section in article.sections]
    else:  # "summary"
        analyzed_summary = nlp.analyze_corpus(article.summary)
        analyzed_summary.idx = 0
        analyzed_summary.name = "summary"
        concepts = analyzed_summary.filter_concepts_by_similarity(dbpedia_confidence)
        concept_locations = Article.get_concept_locations(analyzed_summary.concepts, [analyzed_summary])
        corpus = [analyzed_summary.text]
        # analyzed_summary = []
        # for idx, sentence in enumerate(nlp.analyze_corpus(article.summary).sentences):
        #     analyzed_summary.append({
        #         "idx": idx,
        #         "name": idx,
        #         "sentences": [sentence],
        #         "_concepts": nlp.analyze_corpus(article.summary)._concepts,
        #     })
        # concepts = analyzed_summary.filter_concepts_by_similarity(dbpedia_confidence)
        # concept_locations = Article.get_concept_locations(analyzed_summary.concepts, analyzed_summary)
        # corpus = nlp.analyze_corpus(article.summary).sentences

    concept_tfidfs = nlp.get_concept_tfidf(concepts, corpus)
    results = concept_location_transform(concept_locations, concept_tfidfs)

    # export data for mock server
    # with open("concept_"+mode+".json", "w") as outfile:
    #     json.dump(results, outfile)

    return jsonify(results)


@app.route('/api/concept/sentences', methods=['POST'])
def concept_sentences():
    data = request.get_json()
    title: str = data["title"]
    concepts: Optional[List[str]] = data["concepts"]

    assert isinstance(title, str)
    assert is_list_of_str(concepts) or concepts is None

    article = get_article(title)

    assert article.has_concept(), (
        f"Article: {title} doesn't have concepts.",
        "Please call the API: `concepts` first."
    )

    concepts = concepts or article.concepts
    sentences = article.sentences

    results = {}
    for concept in concepts:
        sent_indices = article.concept_to_sentences(concept, return_idx=True)
        sent_indices = [] if sent_indices is None else sent_indices
        if concept in results:
            results[concept].extend([
                {
                    "sent_idx": sent_idx,
                    "sentence": sentences[sent_idx]
                }
                for sent_idx in sent_indices
            ])
        else:
            results[concept] = [
                {
                    "sent_idx": sent_idx,
                    "sentence": sentences[sent_idx]
                }
                for sent_idx in sent_indices
            ]

    return jsonify(results)


@app.route('/api/concept/cooccurance', methods=['POST'])
def concept_cooccurance():
    data = request.get_json()
    title: str = data["title"]

    article = get_article(title)
    
    # find co-occurence concepts
    co_occurence_concepts = []
    for sentence in article.sentences:
        concepts = article.sentence_to_concepts(sentence, return_idx=False)

        if len(concepts) > 1:  # at least two concepts in a sentence
            co_occurence_concepts.append(concepts)

    # find all pairs
    all_edges = {}
    for entities in co_occurence_concepts:
        edges = combinations(entities, 2)
        for edge in edges:
            edge_repr = f"{edge[0]}::{edge[1]}"
            if edge_repr in all_edges:
                all_edges[edge_repr] += 1
            else:
                all_edges[edge_repr] = 1

    # extract and transform into a list of dicts
    results = []
    for edge_repr, value in all_edges.items():
        source, target = edge_repr.split("::")
        results.append({
            "source": source,
            "target": target,
            "value": value
        })

    return jsonify(results)


@app.route('/api/summary/summarize/<string:mode>', methods=['POST'])
def summarize(mode: Literal["base", "custom", "update"]):
    assert mode in ["base", "custom", "custom_from_base", "update"]

    data = request.get_json()
    title: str = data["title"]
    if mode in ["custom", "custom_from_base", "update"]:
        sentences: List[str] = data["sentences"]
        assert is_list_of_str(sentences)

    article = get_article(title)

    if mode == "base" and (article.summary is None or article.summary_quality is None):
        # if "base" mode and either summary or summary quality is None
        article.base_summary = nlp.summarize(article.text)
        article.base_summary = nlp.clean_up_corpus(article.base_summary)
        article.summary = article.base_summary
    elif mode in ["custom", "custom_from_base"]:
         # custom_text = [article.summary] + sentences
         # custom_text = " ".join(custom_text)
         # article.summary = nlp.summarize(custom_text)
         query = nlp.encode_text(sentences)
         context = article.search_neighbors(query)
         context = " ".join((*context, *sentences)).strip()
         new_summary = nlp.summarize(context, to_shorten_summary=True)
         article.summary = article.summary + " " + new_summary
        # context = " ".join((*article.sentences, *context, *sentences)).strip()
        # prev_summary = article.base_summary if mode == "custom_from_base" else article.summary
        # article.summary = nlp.generate_summary_with_context(context, prev_summary)
    else:  # "update"
        article.summary = " ".join(sentences)

    article.summary = nlp.clean_up_corpus(article.summary)
    article.summary_quality = nlp.rouge_score(article.summary, article.abstract)

    # update article
    global ARTICLES
    ARTICLES[title] = article

    results = {
        "summary": nlp.sentencize(article.summary),
        "summary_quality": article.summary_quality
    }

    return jsonify(results)


@app.route('/api/summary/reorder', methods=['POST'])
def reorder():
    data = request.get_json()
    sentences: List[str] = data["sentences"]

    assert is_list_of_str(sentences)
    
    results = { "sentences": nlp.order_sentences(sentences) }

    return jsonify(results)


@app.route('/api/summary/paraphrase', methods=['POST'])
def paraphrase():
    data = request.get_json()
    sentences: List[str] = data["sentences"]
    num_candidates: Optional[int] = data["num_candidates"]
    num_candidates = num_candidates or 3  # default: return 3 paraphrases for each sentences

    assert is_list_of_str(sentences)
    assert isinstance(num_candidates, int)

    paraphrases = nlp.paraphrase(sentences, num_candidates)
    results = [
        {
            "sentence": sent,
            "paraphrases": list(set(para)),
        }
        for sent, para in zip(sentences, paraphrases)
    ]

    return jsonify(results)


@app.route('/api/embedding/embeddings/<string:mode>', methods=['POST'])
def embeddings(mode: Literal["raw", "project"]):
    assert mode in ["raw", "project"]
    
    data = request.get_json()
    texts: List[str] = data["texts"]
    
    assert is_list_of_str(texts)

    if mode == "project":
        try:
            projection_dims: int = data["projection_dims"]
        except KeyError:
            # projection dimensions' default: 2
            projection_dims = 2

        try:
            projection_mode: Literal["pca", "tsne", "umap"] = data["projection_mode"]
        except KeyError:
            # projection mode's default: "pca"
            projection_mode = "pca"

        assert isinstance(projection_dims, int)
        assert projection_mode in ["pca", "tsne", "umap"]

    embeddings = nlp.encode_text(texts).tolist()
    if mode == "project":
        embeddings = nlp.reduce_dimensions(
            embeddings,
            num_components=projection_dims,
            projection_mode=projection_mode,
            to_list=True
        )

    results = [
        {
            "text": text,
            "embedding": embeddings[idx],
        }
        for idx, text in enumerate(texts)
    ]

    with open("embedding.json", "w") as outfile:
        json.dump(results, outfile)

    return jsonify(results)


@app.route('/api/embedding/similarities', methods=['POST'])
def similarities():
    data = request.get_json()
    sources: List[str] = data["sources"]
    targets: List[str] = data["targets"]

    assert is_list_of_str(sources)
    assert is_list_of_str(targets)
    assert len(sources) == len(targets)

    similarities = nlp.text_similarity(
        sources,
        targets,
        return_list=True,
        return_embeddings=False,
        with_pairs=True
    )["similarities"]

    results = [
        {
            "source": source,
            "target": target,
            "similarity": similarity,
        }
        for source, target, similarity in zip(sources, targets, similarities)
    ]

    return jsonify(results)


@app.route('/api/embedding/focus', methods=['POST'])
def focus():
    data = request.get_json()
    focus_texts: List[str] = data["focus_texts"]
    texts: List[str] = data["texts"]
    projection_dims: Optional[int] = data["projection_dims"]
    projection_mode: Literal["pca", "tsne", "umap"] = data["projection_mode"]
    
    # projection dimensions' default: 2
    projection_dims = 2 if projection_dims is None else projection_dims
    
    assert is_list_of_str(focus_texts)
    assert is_list_of_str(texts)
    assert isinstance(projection_dims, int)
    assert projection_mode in ["pca", "tsne", "umap"]

    use_mean_centroid = len(focus_texts) > 1

    focus_embeddings = nlp.encode_text(focus_texts)
    texts_embeddings = nlp.encode_text(texts)
    centroid_embedding = focus_embeddings.mean(dim=0, keepdim=True) if use_mean_centroid else focus_embeddings
    
    embeddings = [focus_embeddings, texts_embeddings]
    texts = focus_texts + texts  # insert focus_texts at the front
    if use_mean_centroid:
        embeddings.insert(0, centroid_embedding)
        texts.insert(0, None)

    embeddings = torch.cat(embeddings, dim=0)
    proj_embeddings = nlp.reduce_dimensions(
        embeddings.detach().cpu().numpy(),
        num_components=projection_dims,
        projection_mode=projection_mode,
        to_list=True
    )

    similarities = nlp.embedding_distance(centroid_embedding, embeddings).squeeze().tolist()
    
    results = [
        {
            "text": texts[idx],
            "similarity": similarities[idx],
            "proj_embedding": proj_embeddings[idx],
        }
        for idx in range(len(texts))
    ]

    return jsonify(results)


@app.route('/', defaults={'path': ''}, methods=["GET"])
@app.route('/<path:path>')
def index(path):
    '''Return index.html for all non-api routes'''
    #pylint: disable=unused-argument
    return send_from_directory(app.static_folder, 'index.html')