import re
from collections import defaultdict
from typing import Mapping, Union, List, Dict, Literal, Final

import spacy
from nltk.stem import PorterStemmer
import torch
from numpy import ndarray
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from datasets import load_metric
from rouge_score import rouge_scorer

from .multitask_led import MultiTaskLEDConfig, MultiTaskLED
from .article import Section


def clean_up_corpus(corpus: str) -> str:
    """clean_up_corpus

    clean up raw corpus

    Parameters
    ----------
    corpus : str

    Returns
    -------
    str
    """
    # corpus = corpus.lower()
    corpus = re.sub(r"^\s+|\s+$", "", corpus)        # remove leading and trailing spaces
    corpus = re.sub(r"\s\s+", " ", corpus)           # `   ` -> ` ` (remove extra spaces)
    corpus = re.sub(r"\s\,+", ",", corpus)           # ` ,` -> `,`
    corpus = re.sub(r"\s\.+", ".", corpus)           # ` .` -> `.`
    corpus = re.sub(r"\.\.+", ".", corpus)           # `..[...]` -> `.`
    corpus = re.sub(r"\n", "", corpus)
    corpus = re.sub(r"\"", "'", corpus)              # `"` -> `'`
    corpus = re.sub(r"-\s", "-", corpus)             # `- ` -> `-`
    # corpus = re.sub("\([^\)]*,[^\)]*\)", "", corpus)  # remove any text between ()
    # corpus = re.sub("[\(\[].*?[\)\]]", "", corpus)  #
    # corpus = re.sub("^\[[0-9]+(,[0-9]+)*\]$", "", corpus)  # remove [number1, number2, ...]

    return corpus


class NLProcessor(MultiTaskLED):
    """NLProcessor

    A processor for concept extraction, summarization, paraphrasing, 
    sentence similarity, and text embedding tasks.
    """

    # Ref from NLTK
    _stopping_words: Final[List[str]] = [
        "ourselves", "hers", "between", "yourself", "but", "again", "there",
        "about", "once", "during", "out", "very", "having", "with", "they",
        "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into",
        "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who",
        "as", "from", "him", "each", "the", "themselves", "until", "below",
        "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
        "her", "more", "himself", "this", "down", "should", "our", "their", "while",
        "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at",
        "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", 
        "does", "yourselves", "then", "that", "because", "what", "over", "why", 
        "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has",
        "just", "where", "too", "only", "myself", "which", "those", "i", "after",
        "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by",
        "doing", "it", "how", "further", "was", "here", "than", "al", "xiaoyu", 
        "senthil", "yokohama", "15 min", "7 11", "[32", "[39", "[44", "[50"
    ]

    def __init__(
        self,
        config: MultiTaskLEDConfig
    ) -> None:
        super().__init__(config)

        self.entity_extractor = self._get_entitiy_extractor()
        self.stemmer = self._get_stemmer()
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        self.bleu_scorer = load_metric("bleu")

    @property
    def dbpedia_confidence(self) -> float:
        return self.entity_extractor.get_pipe('dbpedia_spotlight').confidence

    @dbpedia_confidence.setter
    def dbpedia_confidence(self, dbpedia_confidence: float) -> None:
        self.entity_extractor.get_pipe('dbpedia_spotlight').confidence = dbpedia_confidence

    @staticmethod
    def clean_up_corpus(corpus: str) -> str:
        return clean_up_corpus(corpus)

    def stem(self, word: str) -> str:
        """stem

        Transforming words to their base form

        Parameters
        ----------
        word : str
            A word

        Returns
        -------
        str
        """
        return self.stemmer.stem(word.lower())

    def sentencize(self, text: str) -> List[str]:
        """sentencize

        Slice a string of text into a list of sentences

        Parameters
        ----------
        text : str
            A string of text

        Returns
        -------
        List[str]
            Sentences
        """
        with self.entity_extractor.select_pipes(disable=['dbpedia_spotlight']):
            doc = self.entity_extractor(text)
            sentences = [self.clean_up_corpus(sent.text) for sent in doc.sents]

        return sentences

    def analyze_corpus(
        self,
        corpus: Union[str, Section],
        dbpedia_confidence: float = 0.1,
        corpus_name: str = "",
    ) -> Section:
        """analyze_corpus

        Clean up corpus, sentencize, extract concepts, and output a Section instance

        Parameters
        ----------
        corpus : Union[str, Section]
            A string of text or a Section instance
        dbpedia_confidence : float, optional
            The threshold for tuning how much confidence the concepts extracted from DBpedia,
            by default 0.1
        corpus_name : str
            The corpus's name. Will be used as Section's name if given, by default ""

        Returns
        -------
        Section
        """
        if isinstance(corpus, Section):
            corpus_name = corpus.name
            corpus = corpus.text

        if dbpedia_confidence != self.dbpedia_confidence:
            self.dbpedia_confidence = dbpedia_confidence

        # clean corpus
        corpus = self.clean_up_corpus(corpus)

        # sentencize, lemmazise, concept entities
        doc = self.entity_extractor(corpus)
        sentences = [sent.text for sent in doc.sents]

        #
        base_to_complex_form = defaultdict(str)
        concept_to_sentence_map = defaultdict(list)
        concept_locations = defaultdict(list)
        concept_similarities = defaultdict(float)
        for ent in doc.ents:
            concept = ent.lemma_
            # replace "-" to " " to avoid same conncepts but in slightly different form, ex: "cross-entropy" -> "cross entropy"
            concept = concept.replace("-", " ")
            concept_base = self.stem(concept)
            if concept in self._stopping_words or concept.isnumeric():
                continue

            base_to_complex_form[concept_base] = concept

            # build concept to sentence map
            sentence_idx = sentences.index(ent.sent.text)
            concept_to_sentence_map[concept_base].append(sentence_idx)
            
            # record concept location
            concept_loc_head = int(ent._.dbpedia_raw_result['@offset'])
            concept_loc_tail = concept_loc_head + len(ent.text)
            concept_location = (concept_loc_head, concept_loc_tail)
            concept_locations[concept_base].append(concept_location)

            # record concept similarities
            concept_simr_score = float(ent._.dbpedia_raw_result['@similarityScore'])
            assert isinstance(concept_simr_score, float)
            concept_similarities[concept_base] = max(concept_similarities[concept_base], concept_simr_score)

        concepts = [concept for concept in concept_to_sentence_map.keys()]  

        return Section(
            name=corpus_name,
            sentences=sentences,
            _concepts=concepts,
            concept_base_to_complex=dict(base_to_complex_form),
            _concept_to_sentence_map=dict(concept_to_sentence_map),
            _concept_locations=dict(concept_locations),
            _concept_similarities=dict(concept_similarities),
        )

    def rouge_score(self, generated: str, reference: str) -> Mapping:
        """rouge_score

        Calculate ROUGE Lsum score

        Parameters
        ----------
        generated : str
            A string of text
        reference : str
            A string of text

        Returns
        -------
        Mapping
            ROUGE score
        """
        
        scores = self.rouge_scorer.score(reference, generated)
        scores = scores['rougeLsum'].fmeasure

        return scores

    def bleu_score(self, source: str, target: str) -> float:
        """bleu_score

        Calculate BLEU score

        Parameters
        ----------
        source : str
            A string of text
        target : str
            A string of text

        Returns
        -------
        float
            BLEU score
        """
        source = self.tokenize_into_string(source)
        target = self.tokenize_into_string(target)

        return self.bleu_scorer.compute(predictions=[source], references=[[target]])["bleu"]

    def order_sentences(self, sentences: List[str], to_concat: bool = False) -> Union[List[str], str]:
        """order_sentences

        Order sentences given their embedding's 1D projection,
        sentences will be ordered from small to large based on embedding values

        Parameters
        ----------
        sentences : List[str]
            A list of sentences
        to_concat : bool, optional
            Whether to concatenate ordered sentences into a string,
            by default False

        Returns
        -------
        Union[List[str], str]
            Ordered sentences or string
        """
        sentence_embeddings = self.encode_text(sentences).cpu().numpy()

        order = KernelPCA(n_components=1, kernel="cosine").fit_transform(sentence_embeddings).flatten()
        order = torch.from_numpy(order)
        order_indices = order.argsort()
        ordered_sentences = [sentences[idx] for idx in order_indices.tolist()]

        if to_concat:
            return " ".join(ordered_sentences)

        return ordered_sentences

    def reduce_dimensions(
        self,
        x: ndarray,
        num_components: int,
        projection_mode: Literal["pca", "tsne", "umap"],
        to_list: bool = False
    ) -> Union[ndarray, List[List[float]]]:
        """reduce_dimensions

        Reduce dimensions using PCA, T-SNE, or UMAP

        Parameters
        ----------
        x : ndarray
            A (n, d) numpy array for projection
        num_components : int
            Number of components (dimensions) to project on to
        projection_mode : Literal[&quot;pca&quot;, &quot;tsne&quot;, &quot;umap&quot;]
            Projection mode
        to_list : bool, optional
            Whether to convert the projected embeddings into list,
            by default False

        Returns
        -------
        Union[ndarray, List[List[float]]]
            Projected embeddings
        """
        assert projection_mode in ["pca", "tsne", "umap"]

        projector = None
        if projection_mode == "pca":
            projector = KernelPCA(num_components, kernel="cosine")
        elif projection_mode == "tsne":
            projector = TSNE(num_components, metric="cosine")
        else:  # "umap"
            projector = UMAP(n_components=num_components, metric="cosine")

        out = projector.fit_transform(x)

        if to_list:
            out = out.tolist()

        return out
            
    @staticmethod
    def get_concept_tfidf(concepts: List[str], corpus: Union[str, List[str]]) -> Dict[str, Mapping]:
        if len(concepts) == 0:
            return {}

        if isinstance(corpus, str):
            corpus = [corpus]

        if len(concepts) > 0:
            max_num_words = max(len(concept.split(" ")) for concept in concepts)
        else:
            max_num_words = 0

        pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=concepts, ngram_range=(1, max_num_words))),
            ('tfid', TfidfTransformer(use_idf=True))
        ]).fit(corpus)
        concept_tfidfs = pipe.transform(corpus).toarray().mean(axis=0).tolist()
        concept_freqs = pipe['count'].transform(corpus).toarray().sum(axis=0).tolist()

        out = {
            concept: {
                "tfidf": tfidf,
                "frequency": freq,
            }
            for concept, tfidf, freq in zip(concepts, concept_tfidfs, concept_freqs)
        }
        
        return out

    @staticmethod
    def _get_entitiy_extractor():
        # only tagger, attribute_ruler, and lemmatizer
        extractor = spacy.load('en_core_web_md', disable=["tok2vec", "ner", "senter", "parser"])

        @spacy.language.Language.component("set_custom_boundaries")
        def set_custom_boundaries(doc):
            for token in doc[: -4]:
                if token.text == "et":  # et al., or et al. (
                    patterns = ["al.,", "al.("]
                    p = "".join(doc[token.i + idx].text for idx in range(1, 4))
                    if p in patterns:
                        doc[token.i + 4].is_sent_start = False
                elif token.text == "(" and doc[token.i + 2].text == ")":  # (xxx)
                    doc[token.i + 1].is_sent_start = False
                elif token.text.isnumeric() and not doc[token.i - 2].text.isnumeric() and doc[token.i - 1].text == "." and doc[token.i + 1].text == ".":  # [a token]. [number].
                    doc[token.i].is_sent_start = True
                    doc[token.i + 2].is_sent_start = False
                elif token.text == ":" and doc[token.i + 1].text.isnumeric() and doc[token.i + 2].text == ".":  # : [number].
                    doc[token.i + 1].is_sent_start = True
                    doc[token.i + 3].is_sent_start = False
                
            return doc

        extractor.add_pipe("sentencizer")
        extractor.add_pipe("set_custom_boundaries")
        # extractor.add_pipe('dbpedia_spotlight', config={'types': None, 'confidence': 0.1})
        extractor.add_pipe('dbpedia_spotlight', config={'types': None, 'confidence': 0.99, 'support': 20})

        return extractor

    @staticmethod
    def _get_stemmer():
        stemmer = PorterStemmer()
        
        # add rules
        irregular_forms = {
            "sky": ["sky", "skies"],
            "die": ["dying"],
            "lie": ["lying"],
            "tie": ["tying"],
            "news": ["news"],
            "inning": ["innings", "inning"],
            "outing": ["outings", "outing"],
            "canning": ["cannings", "canning"],
            "howe": ["howe"],
            "proceed": ["proceed"],
            "exceed": ["exceed"],
            "succeed": ["succeed"],
            "summary": ["summary", "summaries", "summarization", "summarizations"],
            "participant": ["participant", "participants"],
        }
        for key in irregular_forms:
            for val in irregular_forms[key]:
                stemmer.pool[val] = key

        return stemmer