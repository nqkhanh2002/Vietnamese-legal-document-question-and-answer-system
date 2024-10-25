from retrieval.constant import *
from .create_data import *

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, evaluation

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset

import os
os.environ["WANDB_DISABLED"] = "true"

# Create data and load a dataset
model_name = PRETRAIN_MODEL
training_device = DEVICE
model = SentenceTransformer(model_name, device=training_device)

train_path = TRAINING_PATH
test_path = TESTING_PATH
corpus_path = CORPUS_PATH

n_epoch = N_EPOCH
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE

if FAST_DEV_RUN == 0:
    train_df = create_data(train_path=train_path, corpus_path=corpus_path, ran_len=NEGATIVE_NUM)
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    train_dataset = Dataset.from_dict({
        "query": train_df["query"],
        "scope": train_df["article"],
        "label": train_df["label"],
    })

    dev_df = create_data(train_path=test_path, corpus_path=corpus_path, ran_len=NEGATIVE_NUM)
    dev_df = dev_df.sample(frac=1).reset_index(drop=True)

    dev_dataset = Dataset.from_dict({
        "query": dev_df["query"],
        "scope": dev_df["article"],
        "label": dev_df["label"],
    })
else:
    train_df = create_data(train_path=train_path, corpus_path=corpus_path, ran_len=NEGATIVE_NUM)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_df = train_df[:10]

    train_dataset = Dataset.from_dict({
        "query": train_df["query"],
        "scope": train_df["article"],
        "label": train_df["label"],
    })

    dev_df = create_data(train_path=test_path, corpus_path=corpus_path, ran_len=NEGATIVE_NUM)
    dev_df = dev_df.sample(frac=1).reset_index(drop=True)
    dev_df = dev_df[:10]

    dev_dataset = Dataset.from_dict({
        "query": dev_df["query"],
        "scope": dev_df["article"],
        "label": dev_df["label"],
    })
    n_epoch = 1

loss = losses.ContrastiveLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=n_epoch,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    learning_rate=learning_rate,
    eval_strategy="epoch",
    save_strategy="epoch",  
    save_total_limit=5,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    loss=loss,
)
trainer.train()