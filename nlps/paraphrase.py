import warnings
from typing import Final, Literal, Dict, List, Tuple, Mapping

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import (
    LEDForConditionalGeneration,
    LEDTokenizer,
    get_linear_schedule_with_warmup
)
from rouge_score import rouge_scorer


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


class ParaphraseDataset(Dataset):
    """ParaphraseDataset

    A pytorch dataset using `PAWS`, `MRPC`, and `Tapaco` for paraphrasing
    """

    _dataset_split_name: Final[List[str]] = ["train", "validation", "test"]

    def __init__(
        self,
        tokenizer,
        max_len: int = 64,
        mode: Literal["train", "validation", "test"] = "train"
    ) -> None:
        super().__init__()

        assert mode in self._dataset_split_name, (
            f"[{self.__class__.__name__}] `mode` isn't in {self._dataset_split_name}. Got {mode}."
        )
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.sources = None
        self.targets = None
        self._process()

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # tokenize and encode
        source_ids = self.tokenizer(
            self.sources[idx],
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt",
            truncation='longest_first',
        )
        target_ids = self.tokenizer(
            self.targets[idx],
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt",
            truncation='longest_first',
        )

        src_ids = source_ids["input_ids"].squeeze()
        tgt_ids = target_ids["input_ids"].squeeze()

        src_mask = source_ids["attention_mask"].squeeze()
        tgt_mask = target_ids["attention_mask"].squeeze()

        # src_token_type_ids = self.source_ids["token_type_ids"][idx]
        # tgt_token_type_ids = self.target_ids["token_type_ids"][idx]

        return {
            "source_ids": src_ids,
            "target_ids": tgt_ids,
            "source_mask": src_mask,
            "target_mask": tgt_mask,
            # "source_token_type_ids": src_token_type_ids,
            # "target_token_type_ids": tgt_token_type_ids,
        }

    def _process(self) -> None:
        """_process
        load data from Huggingface's dataset and tokenzie
        """
        
        sources = []
        targets = []

        # load PAWS
        sources_, targets_ = self.load_paws(self.mode)
        sources += sources_
        targets += targets_

        # load MRPC
        sources_, targets_ = self.load_mrpc(self.mode)
        sources += sources_
        targets += targets_

        # load tapaco
        sources_, targets_ = self.load_tapaco(self.mode)
        sources += sources_
        targets += targets_

        # clean up spaces
        sources = [self.clean_up_redundant_spaces(src) for src in sources]
        targets = [self.clean_up_redundant_spaces(tgt) for tgt in targets]

        self.sources = sources
        self.targets = targets
        
        assert len(sources) == len(targets), (
            f"[{self.__class__.__name__}] The number of source sentences is not equal to the target one. "
            f"Got {len(sources)} for sources and {len(targets)} for targets."
        )

    def clean_up_redundant_spaces(self, text: str) -> str:
        text = (
            text.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return text

    @staticmethod
    def load_paws(mode) -> Tuple[List[str]]:
        raw_dataset = load_dataset("paws", "labeled_final", split=mode)
        sources = raw_dataset["sentence1"]
        targets = raw_dataset["sentence2"]

        return sources, targets

    @staticmethod
    def load_mrpc(mode) -> Tuple[List[str]]:
        raw_dataset = load_dataset("glue", "mrpc", split=mode)
        labels = raw_dataset["label"]

        sources = []
        targets = []
        for idx, label in enumerate(labels):
            if label == 1:
                sources.append(raw_dataset["sentence1"][idx])
                targets.append(raw_dataset["sentence2"][idx])

        return sources, targets

    @staticmethod
    def load_tapaco(mode) -> Tuple[List[str]]:
        """load_tapaco
        https://huggingface.co/datasets/tapaco/viewer/en/train
        """
        if mode != "train":
            warnings.warn(
                f"Mode is {mode}, but Tapaco dataset only has `train` split, so skip it."
            )
            return [], []
    
        raw_dataset = load_dataset("tapaco", "en", split="train")
        paraphrase_set_indices = raw_dataset["paraphrase_set_id"]
        paraphrases = raw_dataset["paraphrase"]
        paraphrase_set_map = {}
        for idx, para_set_idx in enumerate(paraphrase_set_indices):
            if para_set_idx not in paraphrase_set_map:
                paraphrase_set_map[para_set_idx] = []
            
            paraphrase_set_map[para_set_idx].append(idx)

        sources = []
        targets = []
        for _, paraphrase_indices in paraphrase_set_map.items():
            paraphrase_set = [paraphrases[idx] for idx in paraphrase_indices]
            source = paraphrase_set[0]  # take the first sentence as the source
            target = paraphrase_set[1:]  # the rest as the source's paraphrases
            sources += ([source] * (len(paraphrase_set) - 1))
            targets += target
                
        return sources, targets


class LEDTransferLearnerParaphrase(pl.LightningModule):
    def __init__(self, hparams: Mapping = None) -> None:
        super().__init__()

        if hparams:
            self.hparams.update(hparams)
        self.save_hyperparameters(ignore=['led_cond_gen'])

        self.tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
        # self.model = led_cond_gen.to("cuda")
        self.model = LEDForConditionalGeneration.from_pretrained(
            "allenai/led-large-16384-arxiv",
        ).to("cuda")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None
    ):
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        decoder_input_ids = batch["target_ids"][:, : -1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        # Ref from: https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/models/led/modeling_led.py#L2365-L2368
        labels = batch["target_ids"][:, 1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        # labels = batch["target_ids"][:, 1:].clone()

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.log_metrics({"Loss/Train": loss}, batch_idx)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """validation_step

        Ref from: https://github.com/allenai/longformer/blob/master/scripts/summarization.py#L152-L182
        """
        loss = self._step(batch)
        
        input_ids = batch["source_ids"]
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        generated = self.model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.hparams.max_output_len,
            num_beams=1
        )
        generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        output_ids = batch["target_ids"]
        output_ids[output_ids == -100] = self.tokenizer.pad_token_id
        gold_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        for ref, pred in zip(gold_text, generated_text):
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure

        rouge1 /= len(generated_text)
        rouge2 /= len(generated_text)
        rougel /= len(generated_text)
        rougelsum /= len(generated_text)
        mean_generated_len = torch.Tensor([len(text.split(" ")) for text in generated_text]).mean().item()

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rouge1", rouge1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rouge2", rouge2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rougel", rougel, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rougelsum", rougelsum, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mean_generated_len", mean_generated_len, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.log_metrics({"Loss/Validation": loss}, batch_idx)
        self.logger.log_metrics({"Rouge1/Validation": rouge1}, batch_idx)
        self.logger.log_metrics({"Rouge2/Validation": rouge2}, batch_idx)
        self.logger.log_metrics({"RougeL/Validation": rougel}, batch_idx)
        self.logger.log_metrics({"RougeLsum/Validation": rougelsum}, batch_idx)
        self.logger.log_metrics({"MeanGeneratedLength/Validation": mean_generated_len}, batch_idx)

        return {
            "loss": loss,
            "rouge1": loss.new_zeros(1) + rouge1,
            "rouge2": loss.new_zeros(1) + rouge2,
            "rougel": loss.new_zeros(1) + rougel,
            "rougelsum": loss.new_zeros(1) + rougelsum,
            "mean_generated_len": mean_generated_len
        }
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, on_epoch=True)
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, on_epoch=True)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def freeze_weights(self):
        # fine tune decoder's last two layers
        layer_indices = ["10", "11"]
        for name, param in self.model.named_parameters():
            if "decoder" in name and any(layer_idx in name for layer_idx in layer_indices):
                param.requires_grad = True
            elif "decoder" in name and "layernorm_embedding" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.model.lm_head.weight.requires_grad = True

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        self.freeze_weights()

        parameters = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters.append(param)
                print(f"{name}, {param.shape}")

        optimizer = torch.optim.AdamW(
            parameters,
            # optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        return [optimizer]

    def train_dataloader(self):
        dataset = ParaphraseDataset(tokenizer=self.tokenizer, max_len=self.hparams.max_len, mode="train")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if self.hparams.use_ddp else None
        data_loader = DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        print("Train dataset length:", len(data_loader.dataset))
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
        dataset = ParaphraseDataset(tokenizer=self.tokenizer, max_len=self.hparams.max_len, mode="validation")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.hparams.use_ddp else None
        data_loader = DataLoader(
            dataset,
            batch_size=self.hparams.eval_batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        return data_loader

    def test_dataloader(self):
        dataset = ParaphraseDataset(tokenizer=self.tokenizer, max_len=self.hparams.max_len, mode="test")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.hparams.use_ddp else None
        data_loader = DataLoader(
            dataset,
            batch_size=self.hparams.eval_batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        return data_loader