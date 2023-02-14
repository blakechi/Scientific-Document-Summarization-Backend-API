import os
import gc
import yaml
from math import ceil
from functools import wraps
from dataclasses import dataclass
from typing import (
    Callable,
    Final,
    Tuple,
    List,
    Dict,
    Union
)

import torch
from torch import nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import (
    LEDConfig,
    LEDTokenizer,
    LEDForConditionalGeneration,
)

from .paraphrase import LEDTransferLearnerParaphrase
from .sequence_representation import LEDTransferLearnerSentenceSimilarity


@dataclass
class MultiTaskLEDConfig(object):
    """MultiTaskLEDConfig

    Configuration for MultiTaskLED

    Parameters
    ----------
    text_encoder_max_len : int
        The max token length for the text encoder
    paraphraser_max_len : int
        The max token length for the paraphraser
    summarizer_max_len : int
        The max token length for the summarizer
    text_encoder_path : str
        The path of the checkpoint for the text encoder (sequence representation)
    paraphraser_path : str
        The path of the checkpoint for the paraphraser
    summarizer_path : str
        The path of the checkpoint for the summarizer, by default "allenai/led-large-16384-arxiv"
    tokenizer_path : str
        The path of the tokenizer, by default "allenai/led-large-16384-arxiv"
    device : str
        The device where models exist, will change it to "cpu" if necessary when initialize
        by default "cuda"
    """
    text_encoder_max_len: int
    paraphraser_max_len: int
    summarizer_max_len: int
    text_encoder_path: str
    paraphraser_path: str
    summarizer_path: str = "allenai/led-large-16384-arxiv"
    tokenizer_path: str = "allenai/led-large-16384-arxiv"
    device: str = "cuda"

    @classmethod
    def load(cls, path: os.PathLike) -> "MultiTaskLEDConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls(**config)


def eval_mode(fn: Callable) -> Callable:
    """eval
    Clean all unused memory in the GPUs after inference

    Parameters
    ----------
    fn : Callable
        Function to be wrapped

    Returns
    -------
    Callable
        Wrapped function
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            output = fn(*args, **kwargs)

        gc.collect()
        torch.cuda.empty_cache()

        return output

    return wrapper


class MultiTaskLED(nn.Module):
    """MultiTaskLED

    A multi-task LED for summarization, paraphrasing, text encoding, and sentence similarity
    """
    _tasks: Final[List[str]] = ["summarize", "paraphrase", "encode"]
    _bad_words: Final[List[str]] = [
        "@xcite", "@xmath", "[", "]",
        "* keywords : *", "* Keywords : *",
        "_ keywords : _", "_ Keywords : _", "_ keywords _ :", "_ \n keywords : _"
    ]
    
    def __init__(self, config: MultiTaskLEDConfig) -> None:
        super().__init__()

        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.device == "cuda":
            self.config.device = device

        # tokenizer
        self.tokenizer = self._load_tokenizer()
        self.bad_words_ids = self.tokenizer(self._bad_words, add_prefix_space=True, add_special_tokens=False).input_ids
        # summarizer and encoder
        self.summarizer = self._load_summarizer()
        # text encoder
        self.text_encoder = self._load_text_encoder()
        # paraphraser
        self.paraphraser = self._load_paraphraser()

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        return LEDTokenizer.from_pretrained(self.config.tokenizer_path)

    @eval_mode
    def _load_summarizer(self) -> nn.Module:
        model = LEDForConditionalGeneration.from_pretrained(
            self.config.summarizer_path,  # "allenai/led-large-16384-arxiv"
            return_dict_in_generate=True,
            output_hidden_states=False,
            use_cache=True,
        ).to(self.config.device)

        model.eval()

        return model

    @eval_mode
    def _load_text_encoder(self) -> nn.Module:
        train_module = LEDTransferLearnerSentenceSimilarity.load_from_checkpoint(self.config.text_encoder_path)
        model = train_module.model.to(self.config.device)
        model.config.update({
            "output_hidden_states": False,
            "use_cache": False
        })

        # share the encoder
        setattr(model, "led_encoder", self.get_encoder())

        model.eval()

        return model

    @eval_mode
    def _load_paraphraser(self) -> nn.Module:
        train_module = LEDTransferLearnerParaphrase.load_from_checkpoint(self.config.paraphraser_path)
        model = train_module.model.to(self.config.device)
        model.config.update({
            "output_hidden_states": False,
            "use_cache": True
        })
        
        # share the encoder and partial decoder (except layer 10 and 11, layernorm_embedding, and lm_head)
        setattr(model.led, "encoder", self.get_encoder())
        for layer_idx, layer in enumerate(self.get_decoder().layers):
            if layer_idx != 10 and layer_idx != 11:
                setattr(model.led.decoder.layers, str(layer_idx), layer)

        model.led.set_input_embeddings(model.led.shared)
        
        model.eval()

        return model

    def _share_backbone(self, model: nn.Module) -> nn.Module:
        """_share_backbone

        Share LED's encoder and decoder among all models

        Parameters
        ----------
        model : nn.Module
            A LED model for replacing encoder and decoder

        Returns
        -------
        nn.Module
            A shared model
        """
        setattr(model.led, "encoder", self.get_encoder())
        setattr(model.led, "decoder", self.get_decoder())
        model.led.set_input_embeddings(model.led.shared)  # set encoder's input embeddings back to the original ones

        return model

    def get_encoder(self) -> nn.Module:
        return self.summarizer.get_encoder()

    def get_decoder(self) -> nn.Module:
        return self.summarizer.get_decoder()

    def get_config(self) -> LEDConfig:
        return self.summarizer.config

    def get_global_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """get_global_attention_mask

        A helper function for LED's global attention mask

        Parameters
        ----------
        input_ids : torch.Tensor
            Text's input ids generated from `tokenize`

        Returns
        -------
        torch.Tensor
        """
        mask = torch.zeros_like(input_ids)
        mask[:, 0] = 1  # set global_attention_mask on first token

        return mask

    def tokenize(self, text: List[str], max_length: int) -> Tuple[torch.Tensor]:
        """tokenize

        A helper function for tokenization,
        return input_ids, attention_mask, and global_attention_mask in sequence as a Tuple

        Parameters
        ----------
        text : List[str]
            A list of texts
        max_length : int
            The maximum token length

        Returns
        -------
        Tuple[torch.Tensor]
        """
        output = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            return_tensors="pt",
            truncation='longest_first'
        ).to(self.config.device)

        input_ids = output.input_ids
        attention_mask = output.attention_mask
        global_attention_mask = self.get_global_attention_mask(input_ids)

        return (input_ids, attention_mask, global_attention_mask)

    def tokenize_into_string(self, text: str) -> List[str]:
        encoded = self.tokenizer.tokenize(text)

        return self.tokenizer.convert_tokens_to_string(encoded).split(" ")

    @eval_mode
    def embedding_distance(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        with_pairs: bool = False
    ) -> torch.Tensor:
        """embedding_distance

        Calculate distance between pairs of texts

        Parameters
        ----------
        source : torch.Tensor
            Text embeddings
        target : torch.Tensor
            Text embeddings
        with_pairs : bool
            If True, calculate distance one-to-one (output shape: (n, )),
            if False, calculate distance all-to-all (output shape: (n, m)),
            by default False

        Returns
        -------
        torch.Tensor
        """
        if with_pairs:
            return nn.functional.cosine_similarity(source, target)
        else:
            source_norm = torch.linalg.norm(source, dim=-1, keepdim=True)
            target_norm = torch.linalg.norm(target, dim=-1, keepdim=True)
            cosine_similarity = torch.mm(source, target.t()) / (torch.mm(source_norm, target_norm.t()) + 1e-12)

            return cosine_similarity.squeeze()

    @eval_mode
    def get_encoder_last_hidden_state(self, text: List[str]) -> torch.Tensor:
        """get_encoder_last_hidden_state

        Get the last hidden state of LED's encoder

        Parameters
        ----------
        text : List[str]
            A list of texts

        Returns
        -------
        torch.Tensor
        """
        assert isinstance(text, list)

        input_ids, attention_mask, global_attention_mask = self.tokenize(
            text,
            self.config.summarizer_max_len
        )

        encoder = self.get_encoder()
        last_hidden_state = encoder(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=False,
        ).last_hidden_state.to("cpu")

        return last_hidden_state

    @eval_mode
    def summarize(
        self,
        text: Union[List[str], str],
        to_shorten_summary: bool = False,
    ) -> Union[List[str], str]:
        """summarize

        Summarize a list of texts or a text

        Parameters
        ----------
        text : Union[List[str], str]
            A list of texts or a text

        to_shorten_summary : bool
            Whether to impose length penalty when generating summary,
            by default False

        Returns
        -------
        Union[List[str], str]
        """
        assert isinstance(text, list) or isinstance(text, str)
        
        input_ids, attention_mask, global_attention_mask = self.tokenize(
            text if isinstance(text, list) else [text],
            max_length=self.config.summarizer_max_len
        )

        outputs = self.summarizer.generate(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            max_new_tokens=96 if to_shorten_summary else 1024,
            length_penalty=3. if to_shorten_summary else 1.,
        )
     
        sequences = outputs["sequences"]
        summary = self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return summary if isinstance(text, list) else summary[0]

    @eval_mode
    def generate_summary_with_context(self, context: str, prev_summary: str) -> str:
        """generate_summary_with_context

        Generate new summary that continues after the previous summary given the context.

        Parameters
        ----------
        context : str
            Context as materials for generating new summary
        prev_summary : str
            Previous summary. The new summary will follow what previous summary
            has and generate new text after it

        Returns
        -------
        str
        """
        context_input_ids, context_attention_mask, global_attention_mask = self.tokenize(
            context,
            max_length=self.config.summarizer_max_len
        )

        prev_summary_input_ids = self.tokenizer(prev_summary, return_tensors="pt").input_ids.to("cuda")
        decoder_input_ids = prev_summary_input_ids[:, : -1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        prev_summary_length = decoder_input_ids.shape[-1]

        outputs = self.summarizer.generate(
            context_input_ids,
            attention_mask=context_attention_mask,
            global_attention_mask=global_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            min_length=int(prev_summary_length * 1.5),
            max_new_tokens=prev_summary_length,
            length_penalty=0.7,
            repetition_penalty=3.,
            no_repeat_ngram_size=4,
            bad_words_ids=self.bad_words_ids,
            temperature=1.15,
            top_k=5,
            top_p=0.95,
            do_sample=True,
        )

        sequences = outputs["sequences"]
        summary = self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return summary

    @eval_mode
    def encode_text(
        self,
        text: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """encode_text

        Encode a list of texts

        Parameters
        ----------
        text : List[str]
            A list of texts
        batch_size : int
            Batch size for encoding texts,
            by default 32

        Returns
        -------
        torch.Tensor
        """
        assert isinstance(text, list)

        num_samples = len(text)
        num_batches = ceil(num_samples / batch_size)

        out = []
        for idx in range(num_batches):
            input_ids, attention_mask, global_attention_mask = self.tokenize(
                text[(idx * batch_size): (idx + 1) * batch_size] if idx < (num_batches - 1) else text[(idx * batch_size):],
                self.config.text_encoder_max_len
            )
            encoded = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            out.append(encoded.embedding)

        out = torch.cat(out, dim=0)
        out = nn.functional.normalize(out, p=2, dim=-1)

        return out

    @eval_mode
    def text_similarity(
        self,
        source: Union[List[str], str],
        target: Union[List[str], str],
        batch_size: int = 64,
        return_list: bool = False,
        return_embeddings = False,
        with_pairs: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[float]]]:
        """text_similarity

        Calculate text similarity between source and target.

        Parameters
        ----------
        source : Union[List[str], str]
            A string or a list of text
        target : Union[List[str], str]
            A string or a list of text
        return_list : bool, optional
            Whether to return each pair's similarity and source/target's embeddings as lists,
            if True, will return a list of float,
            if False will return a torch.Tensor,
            by default False.
        return_embeddings : bool 
            Whether to return source's and target's embeddings,
            by default False
        with_pairs : bool
            If True, calculate distance one-to-one (output shape: (n, )),
            if False, calculate distance all-to-all (output shape: (n, m)),
            by default False

        Returns
        -------
        Dict[str, Union[torch.Tensor, List[float]]]
        """
        assert isinstance(source, list) or isinstance(source, str)
        assert isinstance(target, list) or isinstance(target, str)

        source = [source] if isinstance(source, str) else source
        target = [target] if isinstance(target, str) else target

        source_embeddings = self.encode_text(source)
        target_embeddings = self.encode_text(target)

        similarities = self.embedding_distance(source_embeddings, target_embeddings, with_pairs=with_pairs)

        if return_list:
            similarities = similarities.tolist()
        if return_embeddings:
            source_embeddings = source_embeddings.tolist() if return_list else source_embeddings.detach().cpu()
            target_embeddings = target_embeddings.tolist() if return_list else target_embeddings.detach().cpu()
        else:
            source_embeddings = None
            target_embeddings = None
        
        return {
            "similarities": similarities,
            "source_embeddings": source_embeddings,
            "target_embeddings": target_embeddings,
        }

    @eval_mode
    def paraphrase(self, sentences: Union[List[str], str], top_k: int = 3, explore_scale: int = 1) -> List[List[str]]:
        """paraphrase

        Paraphrase a sentence

        Parameters
        ----------
        sentences : Union[List[str], str]
            A list of sentences
        top_k : int, optional
            Return top k paraphrases, by default 3
        explore_scale : int, optional
            The scale of exploration.
            Will explore `top_k`*`explore_scale` of paraphrases,
            by default 1

        Returns
        -------
        List[List[str]]
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        num_sentences = len(sentences)
        num_return_sequences = top_k * explore_scale  # double (or more) the required for later ranking/filtering
        input_ids, attention_mask, global_attention_mask = self.tokenize(
            sentences,
            self.config.paraphraser_max_len
        )

        outputs = self.paraphraser.generate(
            # https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/configuration_utils.py#L123-L183
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            num_return_sequences=num_return_sequences,
            # num_beams=20,
            do_sample=True,
            temperature=2.5,
            top_p=0.5,
            top_k=100,
            repetition_penalty=1.2,
            max_new_tokens=128,
            # length_penalty=1.2,
        )

        paraphrased = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        paraphrased = [
            paraphrased[idx * num_return_sequences: (idx + 1) * num_return_sequences] if idx < (num_sentences - 1) else paraphrased[idx * num_return_sequences:]
            for idx in range(num_sentences)
        ]
        random_sample_indices = torch.randint(0, num_return_sequences, (top_k,))
        paraphrased = [
            [ele[idx] for idx in random_sample_indices]
            for ele in paraphrased
        ]

        return paraphrased

    @classmethod
    def load_from_config_path(cls, path: os.PathLike) -> "MultiTaskLED":
        config = MultiTaskLEDConfig.load(path)
        return cls(config)