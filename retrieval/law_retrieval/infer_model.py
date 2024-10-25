from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import torch.nn as nn

from retrieval.constant import *
from src.train_cross_encoder.create_data import *

import json
import numpy as np
import pandas as pd



model_path = PRETRAIN_MODEL
tokenizer_model = TOKENIZER

device = DEVICE

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

softmax_model = nn.Softmax(dim=1)

def min_max_scale(data, eps = 0.0001):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + eps)

def calculate_recall_precision_f2_score(predicts, labels, threshold):
    predicts = np.argsort(predicts)[::-1]
    predicts = [1 if i <= threshold else 0 for i in range(len(predicts))]
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(predicts)):
        if predicts[i] == 1 and labels[i] == 1:
            tp += 1
        elif predicts[i] == 1 and labels[i] == 0:
            fp += 1
        elif predicts[i] == 0 and labels[i] == 1:
            fn += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    if precision == 0 and recall == 0:
        f2_score = 0
    else:
        f2_score = 5 * precision * recall / (4 * precision + recall)
    return recall, precision, f2_score

def ensemble(df):
    w_bm25_list = []
    w_bert_list = []
    threshold_list = []
    f2_score_list = []
    recall_list = []
    precision_list = []
    for i_bm25 in tqdm(range(0, 11      )):
        w_bm25 = i_bm25 / 10
        w_bert = 1 - w_bm25
        df["ensemble_score"] = w_bm25 * df["bm25_score"] + w_bert * df["bert_score"]
        # min-max scale if have same query
        for query in df["question"].unique():
            query_idx = df[df["question"] == query].index
            df.loc[query_idx, "ensemble_score"] = min_max_scale(df.loc[query_idx, "ensemble_score"].values)
        # arg sort by ensemble score if have same query
        df = df.sort_values(by=["question", "ensemble_score"], ascending=[True, False])
        # get top k score if have same query
        true_list = df[df["label"] == 1].index
        range_list = [1, 2, 5, 10, 20, 50]
        for i_threshold in range_list:
            top_k = i_threshold
            ensemble_score = []
            label = []
            for query in df["question"].unique():
                query_idx = df[df["question"] == query].index
                ensemble_score.extend(df.loc[query_idx, "ensemble_score"].values[:top_k])
                label.extend(df.loc[query_idx, "label"].values[:top_k])
                # count label have 1 or not
                label_count = sum(label)

            recall_list.append(label_count / len(true_list))
            threshold_list.append(top_k)
            w_bm25_list.append(w_bm25)
            w_bert_list.append(w_bert)

    df = pd.DataFrame({
        "w_bm25": w_bm25_list,
        "w_bert": w_bert_list,
        "threshold": threshold_list,
        "recall": recall_list,
    })

    return df

def run_eval(predict_file, output_file):
    predict_df = pd.read_csv(predict_file)

    df = ensemble(predict_df)

    df.to_csv(output_file, index=False)

def min_max_scale(data, eps = 0.0001):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + eps)

def take_predict_label(logits):  # main
    logits = softmax_model(logits)
    logits = logits.tolist()
    label_1_score = []
    for i in logits:
        label_1_score.append(i[1])
    return label_1_score


def model_encode(questions, articles):  # main
    """
    Encode questions and articles
    """
    with torch.no_grad():
        encoded_input = tokenizer(
            questions,
            articles,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)
        output = model(**encoded_input)
        return take_predict_label(output.logits)


def model_batch_encode(questions, articles, batch_size):  # main
    """
    Encode questions and articles in batch size
    """
    predicts = []
    # for i in tqdm(range(0, len(questions), batch_size)):
    for i in tqdm(range(0, len(questions), batch_size)):
        i_start = i
        i_end = i + batch_size
        local_questions = questions[i_start:i_end]
        local_articles = articles[i_start:i_end]
        local_predict = model_encode(local_questions, local_articles)
        predicts.extend(local_predict)
    return predicts

if __name__ == "__main__":      
    test_data = create_multi_case(TESTING_PATH, CORPUS_PATH, NEGATIVE_MODE, NEGATIVE_NUM, 'test')
    if FAST_DEV_RUN == 1:
        test_data = test_data[:10]
    model.eval()
    with torch.no_grad():
        predicts = model_batch_encode(test_data["question"].tolist(), test_data["article"].tolist(), BATCH_SIZE)
        test_data["bert_score"] = predicts
        #min max scale if have same question using group pandas
        test_data["bert_score"] = test_data.groupby("question")["bert_score"].transform(lambda x: min_max_scale(x))
        test_data.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)

    test_data = pd.read_csv(os.path.join(OUTPUT_DIR, "test_data.csv"))
    run_eval(os.path.join(OUTPUT_DIR, "test_data.csv"), os.path.join(OUTPUT_DIR, "ensemble.csv"))