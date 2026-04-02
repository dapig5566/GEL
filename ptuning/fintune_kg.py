#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
# torch.autograd.set_detect_anomaly(True)

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer

from arguments import ModelArguments, DataTrainingArguments
import wandb

logger = logging.getLogger(__name__)


class FB15kIndexMapper:
    special_token_dict = {
        "<SoEB>": 0,  # Start of Entity Block
        "<EoEB>": 1,  # End of Entity Block
        "<AGG>": 2,  # Aggregation Token
        "<MASK>": 3,  # Mask Token
    }

    def __init__(self, entities, relations):
        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2relation = {i: r for r, i in self.relation2id.items()}

    def er2idx(self, entity_or_relation):
        if entity_or_relation in self.entity2id:
            return self.ent2idx(entity_or_relation) + len(
                FB15kIndexMapper.special_token_dict
            )
        elif entity_or_relation in self.relation2id:
            return (
                self.rel2idx(entity_or_relation)
                + len(FB15kIndexMapper.special_token_dict)
                + len(self.entity2id)
            )
        elif entity_or_relation in FB15kIndexMapper.special_token_dict:
            return FB15kIndexMapper.special_token_dict[entity_or_relation]

    def idx2er(self, idx):
        if idx < len(FB15kIndexMapper.special_token_dict):
            return list(FB15kIndexMapper.special_token_dict.keys())[idx]
        elif idx < len(FB15kIndexMapper.special_token_dict) + len(self.entity2id):
            return self.idx2ent(idx - len(FB15kIndexMapper.special_token_dict))
        elif idx < len(FB15kIndexMapper.special_token_dict) + len(self.entity2id) + len(
            self.relation2id
        ):
            return self.idx2rel(
                idx - len(FB15kIndexMapper.special_token_dict) - len(self.entity2id)
            )
        else:
            return None

    def ent2idx(self, entity):
        return self.entity2id.get(entity, -1)

    def idx2ent(self, idx):
        return self.id2entity.get(idx, None)

    def rel2idx(self, relation):
        return self.relation2id.get(relation, -1)

    def idx2rel(self, idx):
        return self.id2relation.get(idx, None)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    config.pre_seq_len = model_args.pre_seq_len
    config.fuse_kg = model_args.fuse_kg
    config.prefix_projection = model_args.prefix_projection
    config.metadata_path = model_args.metadata_path
    config.num_neighbor_samples = model_args.num_neighbor_samples
    config.num_examples = model_args.num_examples
    
    run = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, trust_remote_code=True
        )
        if model_args.pre_seq_len is not None:
            prefix_state_dict = torch.load(
                os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin")
            )
            new_prefix_state_dict = {}
            kb_emb_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder.") :]] = v
                elif k.startswith("transformer.kb_emb."):
                    kb_emb_state_dict[k[len("transformer.kb_emb.") :]] = v

            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            model.transformer.kb_emb.load_state_dict(kb_emb_state_dict, strict=True)
            logger.info(
                f"load checkpoint at `{os.path.join(model_args.ptuning_checkpoint, 'pytorch_model.bin')}` for transformer.kb_emb."
            )
        elif model_args.fuse_kg:
            kg_state_dict = torch.load(
                os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin")
            )
            new_kg_state_dict = {}
            for k, v in kg_state_dict.items():
                if k.startswith("transformer.kg_encoder."):
                    new_kg_state_dict[k[len("transformer.kg_encoder.") :]] = v
            model.transformer.kg_encoder.load_state_dict(
                new_kg_state_dict, strict=False
            )
            logger.info(
                f"checkpoint at `{os.path.join(model_args.ptuning_checkpoint, 'pytorch_model.bin')}` for kg_encoder is loaded."
            )
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, trust_remote_code=True
        )

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
        model.transformer.kb_emb.float()
        model.logger = run
    elif model_args.fuse_kg:
        model = model.half()
        model.transformer.kg_encoder.float()

    else:
        # Finetune
        model = model.float()

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        inputs, targets, infos = [], [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                history = (
                    examples[history_column][i] if history_column is not None else None
                )
                prompt = tokenizer.build_prompt(query, history)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

                info = examples["info"][i]
                if config.pre_seq_len:
                    ids = info.split()
                    c_ids = [int(i) for i in ids]
                    infos.append(c_ids)

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            truncation=True,
            padding=True,
        )
        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )

        if config.pre_seq_len:
            for i, c_ids in enumerate(infos):
                sub_pos = c_ids[:2]
                padded_len = model_inputs["input_ids"][i].count(tokenizer.pad_token_id)
                context_len = len(model_inputs["input_ids"][i]) - padded_len
            model_inputs["infos"] = infos

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

        if config.pre_seq_len:
            model_inputs = {"input_ids": [], "labels": [], "infos": []}
        else:
            model_inputs = {"input_ids": [], "labels": []}

        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                info = examples["info"][i]

                history = (
                    examples[history_column][i] if history_column is not None else None
                )
                prompt = tokenizer.build_prompt(query, history)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(
                    text=prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=data_args.max_source_length,
                )
                b_ids = tokenizer.encode(
                    text=answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=data_args.max_target_length,
                )

                if config.pre_seq_len:
                    ids = info.split()
                    ids = [int(i) for i in ids]
                    c_ids = ids

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = (
                    [tokenizer.pad_token_id] * context_length
                    + b_ids
                    + [tokenizer.eos_token_id]
                )
                
    
                if config.pre_seq_len:
                    task_type = c_ids[0]
                    task_num_subs = {0:1, 1:1, 2:2, 3:5, 4:5}
                    sub_pos = c_ids[2:2+task_num_subs[task_type]*2]

                    for i in range(0, len(sub_pos), 2):
                        for sp in range(sub_pos[i], sub_pos[i + 1] + 1):
                            labels[sp] = tokenizer.pad_token_id
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [
                        (l if l != tokenizer.pad_token_id else -100) for l in labels
                    ]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                if config.pre_seq_len:
                    model_inputs["infos"].append(c_ids)

        return model_inputs

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False,
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print(f"decoded_labels: {decoded_labels}")

        import pickle as pkl

        with open(os.path.join(model_args.ptuning_checkpoint, "mid-output.dat"), "wb") as f:
            pkl.dump((preds, labels, decoded_preds, decoded_labels), f)

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()

            scores = rouge.get_scores(
                " ".join(hypothesis) if len(hypothesis) > 0 else "None",
                " ".join(reference),
            )
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3,
            )
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
        save_changed=model_args.pre_seq_len is not None or model_args.fuse_kg,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if run is not None:
            run.finish()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            do_sample=True,
            top_p=0.7,
            max_length=max_seq_length,
            temperature=0.95,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_seq_length,
            do_sample=True,
            top_p=0.7,
            temperature=0.95,
        )

        import pickle as pkl
        with open(os.path.join(model_args.ptuning_checkpoint, "triple_scores.dat"), "wb") as f:
            pkl.dump(model.triple_scores, f)
            print(f"Save triple scores done: {len(model.triple_scores)}.")

        with open(os.path.join(model_args.ptuning_checkpoint, "attention_scores.dat"), "wb") as f:
            pkl.dump(model.attention_score_list, f)
            print(f"Save attention scores done: {len(model.attention_score_list)}.")
        
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps(
                            {"labels": l, "predict": p}, ensure_ascii=False
                        )
                        writer.write(f"{res}\n")
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
