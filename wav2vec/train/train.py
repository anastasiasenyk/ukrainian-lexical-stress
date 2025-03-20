#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import functools
import json
import logging
import os
import re
import sys
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset, load_from_disk, Audio

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.49.0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt",
)

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        },
    )
    vocab_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to vocab.json"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    feat_proj_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the projected features."}
    )
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
    )
    ctc_loss_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."
        },
    )
    ctc_zero_infinity: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly"
                    " occur when the inputs are too short to be aligned to the targets."
        },
    )
    add_adapter: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether a convolutional attention network should be stacked on top of the Wav2Vec2Bert Encoder. Can be very"
                    "useful to downsample the output length."
        },
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {
                self.feature_extractor_input_name: feature[
                    self.feature_extractor_input_name
                ]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


def setup_dataset(training_args, feature_extractor, tokenizer):
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_from_disk('../data/common_voice_train_data')
    raw_datasets["eval"] = load_from_disk('../data/common_voice_test_data')

    def prepare_dataset(batch):
        audio = batch["audio"]
        # batched output is "un-batched"
        batch["input_values"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        raw_datasets["train"] = raw_datasets["train"].cast_column("audio", Audio(sampling_rate=16_000))
        raw_datasets["eval"] = raw_datasets["eval"].cast_column("audio", Audio(sampling_rate=16_000))

        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            desc="preprocess datasets",
        )

    return vectorized_datasets


def setup_training_with_params(config, model_args, training_args, feature_extractor, tokenizer, vectorized_datasets):
    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "add_adapter": model_args.add_adapter,
        }
    )

    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
    )
    model.freeze_feature_encoder()

    eval_metrics = {
        metric: evaluate.load(metric, cache_dir=model_args.cache_dir)
        for metric in ["wer", "cer"]
    }

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {
            k: v.compute(predictions=pred_str, references=label_str)
            for k, v in eval_metrics.items()
        }
        return metrics

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["eval"],
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # if last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    # elif os.path.isdir(model_args.model_name_or_path):
    #     checkpoint = model_args.model_name_or_path
    # else:
    #     checkpoint = None

    print('\nDEBUG -- checkpoint-- ', training_args.resume_from_checkpoint)

    print('\nDUBUG: training start ')
    print('\nDEBUG -- TRAINING ARGS--\n', training_args)

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    print("\nTRAIN RESULT DEBUG: ", train_result)

    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(vectorized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\nTRAIN METRICS DEBUG: ", metrics)

    # # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(vectorized_datasets["eval"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    trainer.push_to_hub(finetuned_from=model_args.model_name_or_path, )
    print("\nEVAL METRICS DEBUG: ", metrics)

    return metrics['eval_wer']


def main():
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    vocab_file = model_args.vocab_file

    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = training_args.output_dir

        new_vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")
        print("VOCAB FILE PATH: ", new_vocab_file)

        with training_args.main_process_first(desc="dataset map vocabulary creation"):
            if not os.path.isfile(new_vocab_file):
                if vocab_file is not None and os.path.exists(vocab_file):
                    shutil.copy(vocab_file, new_vocab_file)  # Copy the file
                    print(f"Copied {vocab_file} to {new_vocab_file}")
                else:
                    raise FileNotFoundError("vocab.json file does not exist")

        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": (
                config.model_type if config.tokenizer_class is None else None
            ),
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "word_delimiter_token": "|",
        }

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_kwargs,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    vectorized_datasets = setup_dataset(training_args, feature_extractor, tokenizer)

    setup_training_with_params(config, model_args, training_args, feature_extractor, tokenizer, vectorized_datasets)

    return


if __name__ == "__main__":
    main()
