import random
from dataclasses import dataclass
from typing import (
    Callable,
    Type,
    Final,
    Literal,
    Tuple,
    Dict,
    List,
    Mapping,
    Optional,
    Union
)

from numpy import ndarray
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import (
    LEDConfig,
    LEDPreTrainedModel,
    LEDForConditionalGeneration,
    LEDTokenizer,
    get_linear_schedule_with_warmup
)
from transformers.models.led.modeling_led import LEDEncoder
from transformers.utils import ModelOutput


def calculate_train_steps_per_epoch(
    num_train_samples: int,
    train_batch_size: int,
    n_gpu: int,
    gradient_accumulation_steps: int
) -> int:
    steps_per_epoch = (
        num_train_samples // (train_batch_size * max(1, n_gpu))
    ) // gradient_accumulation_steps

    return steps_per_epoch


class STSBDataset(Dataset):
    """STSBDataset

    A pytorch dataset using `stsb` from `glue` for sentence similarity
    """

    _dataset_split_name: Final[List[str]] = ["train", "validation", "test"]

    def __init__(
        self,
        tokenizer,
        max_len: int = 64,
        split: Literal["train", "validation", "test"] = "train",
    ) -> None:
        super().__init__()

        assert split in self._dataset_split_name, (
            f"[{self.__class__.__name__}] `split` isn't in {self._dataset_split_name}. Got {split}."
        )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

        self._process()

    def __len__(self) -> int:
        return len(self.sentence_1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sentence_ids = self.tokenizer.batch_encode_plus(
            [self.sentence_1[idx], self.sentence_2[idx]],
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt",
            truncation='longest_first'
        )
        
        return {
            "input_ids_1": sentence_ids["input_ids"][0],
            "input_ids_2": sentence_ids["input_ids"][1],
            "attention_mask_1": sentence_ids["attention_mask"][0],
            "attention_mask_2": sentence_ids["attention_mask"][1],
            "label": self.labels[idx],
        }

    def _process(self) -> None:
        """_process
        load `stsb` from Huggingface's dataset and tokenzie
        """
        
        sentence_1, sentence_2, labels = self.load_stsb(self.split)

        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.labels = labels

    def load_stsb(self, split: Literal["train", "validate", "test"]) -> Tuple[Union[List[str], List[float]]]:
        raw_dataset = load_dataset("glue", "stsb", split=split)

        sentence_1 = raw_dataset["sentence1"]
        sentence_2 = raw_dataset["sentence2"]
        labels = raw_dataset["label"]
        labels = torch.tensor(labels).to(torch.float32) / 5.  # since it ranges from 0 to 5
        labels = labels.tolist()

        return sentence_1, sentence_2, labels


class AllNLIDataset(Dataset):
    """AllNLIDataset

    A pytorch dataset using `mnli`, `snli` for sentence similarity
    """

    # no "test" due to multi-nli
    _dataset_split_name: Final[List[str]] = ["train", "validation"]

    def __init__(
        self,
        tokenizer,
        max_len: int = 64,
        split: Literal["train", "validation"] = "train",
    ) -> None:
        super().__init__()

        assert split in self._dataset_split_name, (
            f"[{self.__class__.__name__}] `split` isn't in {self._dataset_split_name}. Got {split}."
        )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

        self._process()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        triplet = self.data[idx]
        anchor, positive, negative = triplet
        if random.random() > 0.5:  # switch anchor and positive randomly
            anchor = triplet[1]
            positive = triplet[0]

        triplet_ids = self.tokenizer.batch_encode_plus(
            [anchor, positive, negative],
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt",
            truncation='longest_first',
        )

        return {
            "anchor_ids": triplet_ids["input_ids"][0],
            "positive_ids": triplet_ids["input_ids"][1],
            "negative_ids": triplet_ids["input_ids"][2],
            "anchor_attention_mask": triplet_ids["attention_mask"][0],
            "positive_attention_mask": triplet_ids["attention_mask"][1],
            "negative_attention_mask": triplet_ids["attention_mask"][2],
        }

    def _process(self) -> None:
        """_process
        load `mnli` and `snli` into triplets
        """
        
        mnli_triplets = self.load_mnli(self.split)
        snli_triplets = self.load_snli(self.split)

        self.data = mnli_triplets + snli_triplets

    def load_mnli(self, split: Literal["train", "validation"]) -> List[List[str]]:
        if split == "validation":
            split = "validation_mismatched"

        raw_dataset = load_dataset("multi_nli", "default", split=split)
        sentence_1 = raw_dataset["premise"]
        sentence_2 = raw_dataset["hypothesis"]
        labels = raw_dataset["label"]
        
        triplets = self.make_nli_triplets(sentence_1, sentence_2, labels)

        return triplets

    def load_snli(self, split: Literal["train", "validation", "test"]) -> List[List[str]]:
        raw_dataset = load_dataset("snli", "plain_text", split=split)
        sentence_1 = raw_dataset["premise"]
        sentence_2 = raw_dataset["hypothesis"]
        labels = raw_dataset["label"]
        
        triplets = self.make_nli_triplets(sentence_1, sentence_2, labels)

        return triplets

    @staticmethod
    def make_nli_triplets(sentence_1: List[str], sentence_2: List[str], labels: List[int]) -> List[List[str]]:
        """make_nli_triplets

        Reference from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py#L59-L81

        "entailment": 0
        "neutral": 1
        "contradiction": 2
        """
        grouped_data = {}
        for sent1, sent2, label in zip(sentence_1, sentence_2, labels):
            if sent1 not in grouped_data:
                grouped_data[sent1] = {0: set(), 1: set(), 2: set()}
            
            if label in [0, 1, 2]:
                grouped_data[sent1][label].add(sent2)

        data_triplets = []
        for sent1, others in grouped_data.items():
            if len(others[0]) > 0 and len(others[2]) > 0:
                data_triplets.append([
                    sent1,
                    random.choice(list(others[0])),
                    random.choice(list(others[2]))
                ])
                # skip this and replace by switching anchor and positive randomly (50-50) while training
                # see __getitem__
                # data_triplets.append([
                #     random.choice(list(others[0])),
                #     sent1,
                #     random.choice(list(others[2]))
                # ])
        
        return data_triplets


class MultipleNegativesRankingLoss(nn.Module):
    """MultipleNegativesRankingLoss

    Reference from: https://github.com/UKPLab/sentence-transformers/blob/957c87b3b4cabb96049e9991c7b77624736188af/sentence_transformers/losses/MultipleNegativesRankingLoss.py
    """
    def __init__(self, scale: float = 20.0):
        super().__init__()

        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        sources = anchors
        targets = torch.cat([positives, negatives])

        similarities = self.cosine_similarities(sources, targets) * self.scale
        labels = torch.tensor(range(len(similarities)), dtype=torch.long, device=similarities.device)

        return nn.functional.cross_entropy(similarities, labels)

    def cosine_similarities(self, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sources_norm = nn.functional.normalize(sources, p=2, dim=1)
        targets_norm = nn.functional.normalize(targets, p=2, dim=1)

        return torch.mm(
            sources_norm,
            targets_norm.transpose(0, 1)
        )


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss
    Ref from: https://github.com/UKPLab/sentence-transformers/blob/0462bfa92a144f645dc03a7d1ec625a96a1ab36a/sentence_transformers/losses/CosineSimilarityLoss.py#L1
    """
    def __init__(
        self,
        loss_fct: Callable = nn.MSELoss()
    ) -> torch.Tensor:
        super().__init__()
        
        self.loss_fct = loss_fct
        
    def forward(self, embeddings: torch.Tensor, embeddings_: torch.Tensor, labels: torch.Tensor):
        similarities = torch.cosine_similarity(embeddings, embeddings_, dim=-1)
        loss = self.loss_fct(similarities, labels.view(-1))

        return loss


@dataclass
class LEDSeq2SeqSequenceEmbeddingOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence embedding models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            Cosine embedding loss.
        embedding (`torch.FloatTensor` of shape `(batch_size, config.d_model)`):
    """

    loss: Optional[torch.FloatTensor] = None
    embedding: Optional[torch.FloatTensor] = None


class LEDProjectionHead(nn.Module):
    """
    Head for sentence-level representation learning tasks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_func_name: str = "gelu",
        dropout: float = 0.9,
    ) -> None:
        super().__init__()

        self.hidden_proj = nn.Linear(in_dim, in_dim)
        self.act_func = nn.functional.gelu if act_func_name == "gelu" else getattr(nn, act_func_name)()
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.hidden_proj(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class LEDMeanPooler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # x.shape: B, S, D
        # masks.shape: B, S
        num_elements = masks.sum(dim=-1, keepdims=True)
        out = (x * masks.unsqueeze(dim=-1)).sum(dim=1) / num_elements

        return out


class LEDForSequenceRepresentaion(LEDPreTrainedModel):
    """
    LED model with a sequence projection head on top for representation learning tasks.
    """
    def __init__(self, config: LEDConfig, led_encoder: LEDEncoder, **kwargs):
        super().__init__(config, **kwargs)

        self.led_encoder = led_encoder
        self.mean_pool = LEDMeanPooler()
        self.projection_head = LEDProjectionHead(
            in_dim=config.d_model,
            out_dim=config.d_model,
            act_func_name=config.activation_function,
            dropout=config.dropout
        )

        self.led_encoder._init_weights(self.projection_head.out_proj)
        
    def forward(
        self,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.LongTensor]]] = None,
        global_attention_mask: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        #
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], LEDSeq2SeqSequenceEmbeddingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss = None
        output = None
        if isinstance(input_ids, list):
            tmp = []
            for in_ids, attn_mask in zip(input_ids, attention_mask):
                sequence_embeddings = self._forward(
                    in_ids,
                    attn_mask,
                    global_attention_mask=global_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                tmp.append(sequence_embeddings)
            
            output = torch.stack(tmp)
            anchor_embs, positive_embs, negative_embs = output
            loss_fct = MultipleNegativesRankingLoss()
            loss = loss_fct(anchor_embs, positive_embs, negative_embs)
        else:
            sequence_embeddings = self._forward(
                input_ids,
                attention_mask,
                global_attention_mask=global_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output = sequence_embeddings
            
        if not return_dict:
            # (input_embs, loss) or ((anchor_embs, positive_embs, negative_embs), loss)
            return output, loss

        return LEDSeq2SeqSequenceEmbeddingOutput(
            loss=loss,
            embedding=output,
        )
    
    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:

        output = self.led_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = output.last_hidden_state if return_dict else output[0]
        sequence_embeddings = self.mean_pool(last_hidden_state, attention_mask)
        sequence_embeddings = self.projection_head(sequence_embeddings)
    
        return sequence_embeddings


class LEDTransferLearnerSentenceSimilarity(pl.LightningModule):
    def __init__(
        self,
        led_config: Optional[LEDConfig] = None,
        led_encoder: Optional[LEDEncoder] = None,
        hparams: Optional[Mapping] = None
    ) -> None:
        super().__init__()

        if hparams:
            self.hparams.update(hparams)
        self.save_hyperparameters(ignore=['led_encoder'])

        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
        self.tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
        if led_config is None or led_encoder is None:
            pretrained = LEDForConditionalGeneration.from_pretrained(
                "allenai/led-large-16384-arxiv",
                return_dict_in_generate=True,
            )
            led_config = pretrained.config
            led_encoder = pretrained.get_encoder()
        self.model = LEDForSequenceRepresentaion(led_config, led_encoder)

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        input_ids_ = input_ids[0] if isinstance(input_ids, list) else input_ids
        global_attention_mask = torch.zeros_like(input_ids_)
        global_attention_mask[:, 0] = 1

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        convert_to_numpy: bool = False,
        to_normalize: bool = False,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, ndarray]:
        self.model.eval()
        
        out = []
        num_batches = (len(texts) // batch_size) + 1
        for batch_idx in range(num_batches):
            with torch.no_grad():
                batch_texts = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size] if (batch_idx + 1) < num_batches else texts[batch_idx * batch_size:]
                outputs = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    padding='max_length',
                    return_tensors="pt",
                    truncation='longest_first',
                )
                input_ids = outputs["input_ids"]
                attention_mask = outputs["attention_mask"]
                outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.embedding

                out.append(embeddings)

        out = torch.cat(embeddings).to("cpu")
        if to_normalize:
            out = nn.functional.normalize(out, p=2, dim=-1)

        if convert_to_numpy:
            out = out.numpy()

        return out

    def _step(self, batch):
        anchor_ids = batch["anchor_ids"]
        positive_ids = batch["positive_ids"]
        negative_ids = batch["negative_ids"]
        anchor_attention_mask = batch["anchor_attention_mask"]
        positive_attention_mask = batch["positive_attention_mask"]
        negative_attention_mask = batch["negative_attention_mask"]
        outputs = self(
            input_ids=[anchor_ids, positive_ids, negative_ids],
            attention_mask=[anchor_attention_mask, positive_attention_mask, negative_attention_mask],
        )

        loss = outputs.loss
        embedding = outputs.embedding

        return loss, embedding

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.log_metrics({"Loss/Train": loss}, batch_idx)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.log_metrics({"Loss/Validation": loss}, batch_idx)

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, on_epoch=True)
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, on_epoch=True)

    def freeze_weights(self):
        for name, param in self.model.named_parameters():
            if "projection_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        self.freeze_weights()

        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        return [optimizer]

    def train_dataloader(self):
        train_dataset = self.get_dataset("train")
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
        
        train_steps_per_epoch = calculate_train_steps_per_epoch(
            len(data_loader.dataset),
            self.hparams.train_batch_size,
            self.hparams.n_gpu,
            self.hparams.gradient_accumulation_steps
        )
        total_steps = train_steps_per_epoch * self.hparams.num_train_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps
        )
        
        return data_loader

    def val_dataloader(self):
        val_dataset = self.get_dataset("validation")
        data_loader = DataLoader(
            val_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

        return data_loader

    def get_dataset(self, split: str) -> Type[AllNLIDataset]:
        return AllNLIDataset(
            tokenizer=self.tokenizer,
            max_len=self.hparams.max_len,
            split=split,
        )
