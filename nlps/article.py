from itertools import chain
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Type, Iterable, Dict, List, Tuple, Optional, Union, Mapping

import numpy as np
import torch
import faiss


@dataclass
class Section(object):
    """Section
    Data holder for sections
    """
    name: str = field(default_factory=str)
    idx: Optional[int] = None
    sentences: List[str] = field(default_factory=list)
    # sentence_paraphrases: Optional[Dict[int, List[str]]] = None
    _concepts: List[str] = field(default_factory=list)
    concept_base_to_complex: Optional[Dict[str, str]] = None
    # _concept_embeddings: Optional[Dict[str, torch.Tensor]] = None
    _concept_to_sentence_map: Optional[Dict[str, List[int]]] = None
    _concept_locations: Optional[Dict[str, List[Tuple[int, int]]]] = None
    _concept_similarities: Optional[Dict[str, float]] = None

    @property
    def concepts(self) -> List[str]:
        if self.concept_base_to_complex is not None:
            return [self.concept_base_to_complex[concept] for concept in self._concepts]

        return self._concepts

    @concepts.setter
    def concepts(self, concepts: List[str]) -> None:
        self._concepts = concepts

    @property
    def concept_to_sentence_map(self) -> Optional[Dict[str, List[int]]]:
        if self.concept_base_to_complex is not None and self._concept_to_sentence_map is not None:
            return {
                self.concept_base_to_complex[concept]: value
                for concept, value in self._concept_to_sentence_map.items()
            }

        return self._concept_to_sentence_map

    @concept_to_sentence_map.setter
    def concept_to_sentence_map(self, concept_to_sentence_map: Optional[Dict[str, List[int]]]) -> None:
        self._concept_to_sentence_map = concept_to_sentence_map

    @property
    def concept_locations(self) -> Optional[Dict[str, List[Tuple[int, int]]]]:
        if self.concept_base_to_complex is not None and self._concept_locations is not None:
            return {
                self.concept_base_to_complex[concept]: value
                for concept, value in self._concept_locations.items()
            }

        return self._concept_locations

    @concept_locations.setter
    def concept_locations(self, concept_locations: Optional[Dict[str, List[Tuple[int, int]]]]) -> None:
        self._concept_locations = concept_locations

    @property
    def concept_similarities(self) -> Optional[Dict[str, float]]:
        if self.concept_base_to_complex is not None and self._concept_similarities is not None:
            return {
                self.concept_base_to_complex[concept]: value
                for concept, value in self._concept_similarities.items()
            }

        return self._concept_similarities

    @concept_similarities.setter
    def concept_similarities(self, concept_similarities: Optional[Dict[str, float]]) -> None:
        self._concept_similarities = concept_similarities

    @property
    def text(self) -> str:
        """text
        Generate the section's full text by concatenating sentences

        Returns
        -------
        str
        """
        return (" ".join(self.sentences)).strip()

    def get_sentence_idx(self, sentence: str) -> Optional[int]:
        """get_sentence_idx
        Return local sentence index in the section given a sentence,
        if the sentence doesn't exist, return None

        Parameters
        ----------
        sentence : str
            sentence text

        Returns
        -------
        Optional[int]
        """
        try:
            return self.sentences.index(sentence)
        except ValueError:
            return None

    def filter_concepts_by_similarity(self, similarity: float) -> List[str]:
        return [concept for concept, simr in self.concept_similarities.items() if simr >= similarity]

    def __len__(self) -> int:
        return len(self.sentences)


@dataclass
class Article(object):
    """Article
    Data holder for articles
    """
    title: Optional[str] = None
    abstract: Optional[str] = None
    _sections: Optional[List[Type[Section]]] = None
    concept_base_to_complex: Optional[Dict[str, str]] = None
    _sentence_embeddings: Optional[torch.Tensor] = None
    _search_index: Optional[faiss.IndexFlatIP] = None
    base_summary: Optional[str] = None
    summary: Optional[str] = None
    summary_quality: Optional[float] = None

    @property
    def sections(self) -> Optional[List[Section]]:
        return self._sections

    @sections.setter
    def sections(self, sections: List[Section]) -> None:
        global_concept_base_to_complex = {}
        for section in sections:
            if section.concept_base_to_complex is not None:
                global_concept_base_to_complex.update(section.concept_base_to_complex)
        
        # add section index
        for idx, section in enumerate(sections):
            section.idx = idx
            section.concept_base_to_complex = global_concept_base_to_complex

        self._sections = sections
        self.concept_base_to_complex = global_concept_base_to_complex
        
    @property
    def sentence_embeddings(self) -> Optional[torch.Tensor]:
        return self._sentence_embeddings

    @sentence_embeddings.setter
    def sentence_embeddings(self, sentence_embeddings: torch.Tensor) -> None:
        # assert isinstance(sentence_embeddings, torch.Tensor)

        _, dim = sentence_embeddings.shape
        search_index = faiss.IndexFlatIP(dim)
        search_index.add(sentence_embeddings)

        self._search_index = search_index
        self._sentence_embeddings = sentence_embeddings

    @property
    def sentences(self) -> List[str]:
        if self.has_section():
            return list(chain.from_iterable(section.sentences for section in self.sections))

        return []

    @property
    def concepts(self) -> List[str]:
        if self.has_section():
            # concatenate, remove duplicates, to list
            return list(OrderedDict.fromkeys(chain.from_iterable(section.concepts for section in self.sections)))
        
        return []

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)
    
    @property
    def text(self) -> str:
        """text
        Generate the article's full text by concatenating sections

        Returns
        -------
        str
        """
        if self.has_section():
            return (" ".join(section.text for section in self.sections)).strip()

        return ""

    @property
    def concept_locations(self) -> Optional[List[Mapping]]:
        """concept_locations
        Return concepts' locations in each section

        Returns
        -------
        Optional[List[Mapping]]
        """
        if self.has_section():
            global_concepts = self.concepts
            
            return self.get_concept_locations(global_concepts, self.sections)

        return None

    def has_section(self, section: Optional[Union[int, str]] = None) -> bool:
        """has_section
        Whether the article has any sections or the section

        Parameters
        ----------
        section : Optional[Union[int, str]], optional
            None (if there is any), section name, or section index, by default None

        Returns
        -------
        bool
        """
        if self.sections is not None:
            if section is None:
                return True
            elif isinstance(section, int):  # section index
                return 0 <= section and section < len(self)
            else:
                return section in self.get_section_names()
        
        return False

    def has_sentence(self, sentence: Optional[Union[int, str]] = None) -> bool:
        """has_sentence
        Whether the article has any sentences or the sentence

        Parameters
        ----------
        section : Optional[Union[int, str]], optional
            None (if there is any), sentence text, or sentence index, by default None

        Returns
        -------
        bool
        """
        if sentence is None:
            return self.num_sentences > 0

        if self.has_section():
            if isinstance(sentence, int):  # sentence index
                return 0 <= sentence and sentence < self.num_sentences
            else:
                return any(
                    section.get_sentence_idx(sentence) is not None
                    for section in self.sections
                )

        return False

    def has_concept(self, concept: Optional[Union[int, str]] = None) -> bool:
        """has_concept
        Whether the article has any concepts or the concepts

        Parameters
        ----------
        section : Optional[Union[int, str]], optional
            None (if there is any), concept text, or concept index, by default None

        Returns
        -------
        bool
        """
        if concept is None:
            return self.num_concepts > 0

        if self.concepts is not None:
            if isinstance(concept, int):  # concept index
                return 0 <= concept and concept < len(self.concepts)
            else:
                return concept in self.concepts
        
        return False

    def get_section_names(self, to_list: bool = False) -> Optional[Union[List[str], Iterable]]:
        """get_section_names
        Return None when there are no sections, else return all section names

        Parameters
        ----------
        to_list : bool
            whether to convert to list, by default False

        Returns
        -------
        Optional[Union[List[str], Iterable]]
        """
        if self.has_section():
            if to_list:
                return [section.name for section in self.sections]
            else:
                return iter(section.name for section in self.sections)
        
        return None

    def get_section_name(self, section_idx: int) -> Optional[str]:
        """get_section_name
        Return None when there are no sections, else return one section name you specify

        Parameters
        ----------
        section_idx : int
            section index

        Returns
        -------
        Optional[str]
        """
        if self.has_section(section_idx):
            return self.get_section_names(to_list=True)[section_idx]

        return None

    def filter_concepts_by_similarity(self, similarity: float) -> List[str]:
        if self.has_section():
            concepts = list(chain.from_iterable(
                section.filter_concepts_by_similarity(similarity)
                for section in self.sections
            ))
            concepts = list(OrderedDict.fromkeys(concepts))  # remove duplicated
            
            return concepts

        return []

    def _get_sentence_idx_offset_by_sections(self) -> Optional[List[int]]:
        """_get_sentence_idx_offset_by_sections
        Return sentence index offset for each sections,
        mainly for converting section-wise sentence indices to article-wise indices

        Returns
        -------
        Optional[List[int]]
        """
        if self.has_section():
            sentence_idx_offsets = [0]
            acc_offset = 0
            for section in self.sections[: -1]:
                acc_offset += len(section)
                sentence_idx_offsets.append(acc_offset)

            return sentence_idx_offsets

        return None

    def search_neighbors(
        self,
        query: torch.Tensor,
        num_neighbors: int = 10,
        to_sort: bool = False,
        return_idx: bool = False
    ) -> Optional[Union[List[str], List[int]]]:
        query = query.cpu().detach().numpy()
        _, sentence_indices = self._search_index.search(query, num_neighbors)
        sentence_indices = np.unique(sentence_indices.flatten())
        if to_sort:
            sentence_indices = np.sort(sentence_indices)
        
        if return_idx:
            return sentence_indices

        _sentences = self.sentences
        searched = [_sentences[idx] for idx in sentence_indices]

        return searched

    def concept_to_sentences(
        self,
        concept: str,
        return_idx: bool = False,
    ) -> Optional[Union[List[str], List[int]]]:
        """concept_to_sentences
        Given a concept in Article, return a list of sentences that contain the concept,
        return None if the concept doesn't exist

        Parameters
        ----------
        concept : str
            concept appears in the article
        return_idx : bool, optional
            return concept index if True, else return concept text, by default False

        Returns
        -------
        Optional[Union[List[str], List[int]]]
        """
        if self.has_concept(concept):
            output = []
            for section, sentence_offset in zip(self.sections, self._get_sentence_idx_offset_by_sections()):
                if concept in section.concepts:
                    local_sentence_indices = section.concept_to_sentence_map[concept]
                    if return_idx:
                        output = output + [idx + sentence_offset for idx in local_sentence_indices]
                    else:
                        output = output + [section.sentences[idx] for idx in local_sentence_indices]
            
            return output

        return None

    def sentence_to_concepts(
        self,
        sentence: str,
        return_idx: bool = False,
    ) -> Optional[Union[List[str], List[int]]]:
        """sentence_to_concepts
        Given a sentence in Article, return a list of concepts that appear in the sentence,
        return None if the sentence doesn't exist

        Parameters
        ----------
        sentence : str
            sentence appears in the article
        return_idx : bool, optional
            return sentence index if True, else return sentence text, by default False

        Returns
        -------
        Optional[Union[List[str], List[int]]]
        """
        if self.has_sentence(sentence):
            section_idx = self.sentence_to_section(sentence, return_idx=True)
            section = self.sections[section_idx]
            sentence_idx = section.get_sentence_idx(sentence)

            concepts = [
                concept
                for concept, sentence_indices in section.concept_to_sentence_map.items()
                if sentence_idx in sentence_indices
            ]

            if return_idx:
                global_concepts = self.concepts

                return [global_concepts.index(concept) for concept in concepts]
            
            return concepts
        
        return None

    def section_to_sentences(
        self,
        section: str,
        return_idx: bool = False,
    ) -> Optional[Union[List[str], List[int]]]:
        """section_to_sentences
        Given a section title in Article, return a list of sentences in the section,
        return None if the section doesn't exist

        Parameters
        ----------
        section : str
            section title in the article
        return_idx : bool, optional
            return sentence index if True, else return sentence text, by default False

        Returns
        -------
        Optional[Union[List[str], List[int]]]
        """
        if self.has_section(section):
            section_idx = self.get_section_names(to_list=True).index(section)
            sentences = self.sections[section_idx].sentences

            if return_idx:
                sentence_offset = self._get_sentence_idx_offset_by_sections()[section_idx]
                
                return [idx + sentence_offset for idx in range(len(sentences))]

            return sentences

        return None

    def sentence_to_section(
        self,
        sentence: str,
        return_idx: bool = False,
    ) -> Optional[Union[str, int]]:
        """sentence_to_section
        Given a sentence in Article, return the corresponding section name,
        return None if the sentence doesn't exist

        Parameters
        ----------
        sentence : str
            sentence text
        return_idx : bool, optional
            return section index if True, else return section text, by default False

        Returns
        -------
        Optional[Union[str, int]]
        """
        if self.has_sentence(sentence):
            for section_idx, section in enumerate(self.sections):
                local_sentence_idx = section.get_sentence_idx(sentence)

                if local_sentence_idx is not None:
                    return section_idx if return_idx else self.sections[section_idx].name

        return None

    def section_to_concepts(
        self,
        section: str,
        return_idx: bool = False,
    ) -> Optional[Union[List[str], List[int]]]:
        """section_to_concepts
        Given a section title in Article, return the concepts appear in that section,
        return None if the section doesn't exist

        Parameters
        ----------
        section : str
            section title
        return_idx : bool, optional
            return concept index if True, else return concept text, by default False

        Returns
        -------
        Optional[Union[List[str], List[int]]]
        """
        if self.has_section(section):
            section_idx = self.get_section_names(to_list=True).index(section)
            concepts = self.sections[section_idx].concepts

            if return_idx:
                global_concepts = self.concepts

                return [global_concepts.index(concept) for concept in concepts]
            else:
                return concepts

        return None

    def concept_to_sections(
        self,
        concept: str,
        return_idx: bool = False,
    ) -> Optional[Union[List[str], List[int]]]:
        """concept_to_sections
        Given a concept in Article, return the section names that contains that concept,
        return None if the concept doesn't exist

        Parameters
        ----------
        concept : str
            concept text
        return_idx : bool, optional
            return section index if True, else return section text, by default False

        Returns
        -------
        Optional[Union[List[str], List[int]]]
        """
        if self.has_concept(concept):
            section_indices = [
                section_idx
                for section_idx, section in enumerate(self.sections)
                if concept in section.concepts
            ]

            if return_idx:
                return section_indices
            else:
                return [self.sections[idx].name for idx in section_indices]
            
        return None

    @staticmethod
    def get_concept_locations(all_concepts: List[str], sections: List[Section]) -> List[Mapping]:
        """get_concept_locations

        Given a list of concepts and sections, return concepts' locations in sections

        Parameters
        ----------
        all_concepts : List[str]
            A list of concepts
        sections : List[Section]
            A list of sections

        Returns
        -------
        List[Mapping]
        """
        output = []
        for section in sections:
            concepts = []
            for idx, concept in enumerate(all_concepts):
                if concept in section.concept_locations:
                    concepts.append({
                        "concept": concept,
                        "concept_idx": idx,
                        "locations": section.concept_locations[concept]
                    })

            output.append({
                "section": section.name,
                "section_idx": section.idx,
                "concepts": concepts
            })

        return output

    def update(self, other: "Article") -> None:
        """update
        In-place update given an `Article`, only update the fields that are not None in `other`

        Parameters
        ----------
        other : Article
        """
        self += other

    def __iadd__(self, other: "Article") -> "Article":
        for field in fields(other):
            other_value = getattr(other, field.name)
            if other_value is not None:
                setattr(self, field.name, other_value)

        return self

    def __len__(self) -> int:
        return len(self.sections) if self.has_section() else 0