from retrieval.constant import *
from .create_data import *

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np

import pandas as pd


def read_data(train_path, test_path):
    with open(train_path, "r") as f:
        question_data_train = json.load(f)
    with open(test_path, "r") as f:
        question_data_test = json.load(f)
    return question_data_train, question_data_test


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["inputs"], max_length=1024, truncation=True, padding=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["labels"], max_length=10, truncation=True, padding=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_training_data_17k(question_data):
    inputs = []
    labels = []
    for question in question_data:
        for relevant_law in question["relevant_laws"]:
            query = question["question"]
            document = relevant_law["content"]

            soft_negative = relevant_law["soft_negative"]
            hard_negative = relevant_law["hard_negative"]

            inputs.append(f"Câu hỏi: {query} Điều luật: {document} Có liên quan:")
            labels.append("đúng")

            for sn in soft_negative:
                inputs.append(f"Câu hỏi: {query} Điều luật: {sn} Có liên quan:")
                labels.append("sai")

            try:
                for hn in hard_negative:
                    inputs.append(f"Câu hỏi: {query} Điều luật: {hn} Có liên quan:")
                    labels.append("sai")
            except:
                print(hard_negative)
    return inputs, labels

def create_training_data_alqac(df_data):
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    inputs = []
    labels = []
    query = df_data['question_list']
    article = df_data['article_list']
    label = df_data['label_list']
    for i in range(len(query)):
        local_query = query[i]
        local_article = article[i]
        local_label = label[i]
        inputs.append(f"Câu hỏi: {local_query} Điều luật: {local_article} Có liên quan:")
        if local_label == 1:
            labels.append("đúng")
        else:
            labels.append("sai")
    return inputs, labels

def create_training_data(question_data):
    if MODEL_TYPE == "17k":
        return create_training_data_17k(question_data)
    elif MODEL_TYPE == "ALQAC":
        return create_training_data_alqac(question_data)

model_name = PRETRAIN_MODEL
tokenizer_name = TOKENIZER
training_device = DEVICE

train_path = TRAINING_PATH
test_path = TESTING_PATH
corpus_path = CORPUS_PATH

n_epoch = N_EPOCH
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE

if MODEL_TYPE == "17k":
    # Load the question and law data
    with open(TRAINING_PATH, 'r', encoding="utf-8") as qftr:
        question_data_train = json.load(qftr)
            
    with open(TESTING_PATH, 'r', encoding="utf-8") as qfte:
        question_data_test = json.load(qfte)

    with open(CORPUS_PATH, 'r', encoding="utf-8") as lf:
        law_data = json.load(lf)

    law_dict = {}
    for law in law_data:
        law_dict[law['id']] = law

    train_set = process_questions(question_data_train, law_dict)
    test_set = process_questions(question_data_test, law_dict)
elif MODEL_TYPE == "ALQAC":
    question_data_train = pd.read_csv(TRAINING_PATH)
    question_data_test = pd.read_csv(TESTING_PATH)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(DEVICE)

if FAST_DEV_RUN == 1:
    question_data_train = question_data_train[:10]
    question_data_test = question_data_test[:10]
    train_inputs, train_labels = create_training_data(question_data_train)
    train_dict_obj = {"inputs": train_inputs, "labels": train_labels}
    train_dataset = Dataset.from_dict(train_dict_obj)
    train_tokenized_datasets = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["inputs"], num_proc=8
    )
    test_inputs, test_labels = create_training_data(question_data_test)
    test_dict_obj = {"inputs": test_inputs, "labels": test_labels}
    test_dataset = Dataset.from_dict(test_dict_obj)
    test_tokenized_datasets = test_dataset.map(
        preprocess_function, batched=True, remove_columns=["inputs"], num_proc=8
    )
else:

    # question_data_train, question_data_test = read_data(TRAINING_PATH, TESTING_PATH)
    # Example usage with the question_data
    train_inputs, train_labels = create_training_data(question_data_train)
    train_dict_obj = {"inputs": train_inputs, "labels": train_labels}
    train_dataset = Dataset.from_dict(train_dict_obj)
    train_tokenized_datasets = train_dataset.map(
        preprocess_function, batched=True, remove_columns=["inputs"], num_proc=8
    )

    test_inputs, test_labels = create_training_data(question_data_test)
    test_dict_obj = {"inputs": test_inputs, "labels": test_labels}
    test_dataset = Dataset.from_dict(test_dict_obj)
    test_tokenized_datasets = test_dataset.map(
        preprocess_function, batched=True, remove_columns=["inputs"], num_proc=8
    )

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

# Training arguments - OUT OF MEM
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=n_epoch,
    learning_rate=learning_rate,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    group_by_length=True,
    eval_strategy="epoch",
    save_strategy="epoch",  
    save_total_limit=5,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()