from retrieval.constant import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import pandas as pd
from datasets import Dataset, load_metric

model_name = PRETRAIN_MODEL
tokenizer_name = TOKENIZER
training_device = DEVICE
file_path = TESTING_PATH
output_dir = OUTPUT_DIR
output_path = OUTPUT_PATH

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAIN_MODEL)
model.to(DEVICE)
model.eval()

public_test_df = pd.read_csv(file_path)


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


# Prepare your test data
input_lines = []
label_lines = []
data = []

for index, row in public_test_df.iterrows():
    query = row["query"]
    article_id = row["article_id"]
    document = row["content"]
    actual_label = "đúng" if row["label"] else "sai"

    input_lines.append(
        f"Câu hỏi: {query} Điều luật: Điều {article_id}. {document} Có liên quan:"
    )
    label_lines.append(actual_label)
    data.append({"question": query, "document": document, "actual_label": actual_label})

# Create DataFrame
df = pd.DataFrame(data)

input_lines = input_lines
label_lines = label_lines

dict_obj = {"inputs": input_lines, "labels": label_lines}
# dict_obj = {'inputs': input_lines}

# def preprocess_function_2(examples):
#     model_inputs = tokenizer(
#         examples["inputs"], max_length=512, truncation=True, padding=True
#     )
#     return model_inputs

dataset = Dataset.from_dict(dict_obj)
test_tokenized_datasets = dataset.map(
    preprocess_function, batched=True, remove_columns=["inputs"], num_proc=10
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

# Cach 1: Duoc nhg ra score k lien quan cho lam
bs = 32
dataloader = torch.utils.data.DataLoader(
    test_tokenized_datasets, collate_fn=data_collator, batch_size=bs
)
max_target_length = 10

predictions = []
references = []

prob_of_token_true = []
most_likely_token_probss = []

model.eval()

for i, batch in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        b_size = len(batch["input_ids"])

        outputs = model.generate(
            input_ids=batch["input_ids"].to("cuda"),
            max_length=max_target_length,
            attention_mask=batch["attention_mask"].to("cuda"),
        )

        decoder_input_ids = torch.tensor(
            [[1]] * b_size
        )  # Initialize decoder input IDs with a bos token (1)
        decoder_attention_mask = torch.tensor(
            [[1]] * b_size
        )  # Initialize decoder attention mask

        logits = model(
            input_ids=batch["input_ids"].to("cuda"),
            attention_mask=batch["attention_mask"].to("cuda"),
            decoder_input_ids=decoder_input_ids.to("cuda"),
            decoder_attention_mask=decoder_attention_mask.to("cuda"),
        )[0]
        probs_batch = torch.exp(torch.log_softmax(logits, dim=-1))
        #         print(probs_batch)
        #         print(probs_batch.shape)

        # lưu cái đống prob của "đúng"
        prob_of_token_true.extend(probs_batch[:, 0, 1356].tolist())

        most_likely_token_index = torch.argmax(probs_batch, dim=-1)
        most_likely_token_probs = torch.max(probs_batch, dim=-1)[0]
        most_likely_token_probss.extend(most_likely_token_probs.tolist())
        #         print(most_likely_token_probs.tolist())

        with tokenizer.as_target_tokenizer():
            outputs = [
                tokenizer.decode(
                    out, clean_up_tokenization_spaces=False, skip_special_tokens=True
                )
                for out in outputs
            ]
            labels = np.where(
                batch["labels"] != -100, batch["labels"], tokenizer.pad_token_id
            )
            actuals = [
                tokenizer.decode(
                    out, clean_up_tokenization_spaces=False, skip_special_tokens=True
                )
                for out in labels
            ]
        predictions.extend(outputs)
        references.extend(actuals)

public_test_df["ViMonoT5_score"] = prob_of_token_true
public_test_df["ViMonoT5_predict"] = [
    1 if value == "đúng" else 0 for value in predictions
]

public_test_df.to_csv(os.path.join(output_dir, output_path), index=False)
