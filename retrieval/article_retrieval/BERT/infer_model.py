from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import random
import pandas as pd

from retrieval.constant import *

import warnings
warnings.filterwarnings('ignore')

tf.keras.utils.disable_interactive_logging()

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def model_encoding(query: str):
    query_embedding = model.encode(query, device=device, show_progress_bar = False)
    return query_embedding

def batch_encoding(query_list: list):
    query_embedding = model.encode(query_list, device=device, show_progress_bar = False)
    return query_embedding.tolist()

def infer_law_list(data_list, batch_mode = True):
    case_list = []
    for i in tqdm(range(len(data_list))):
        i_case = data_list[i]
        para = i_case["paragraph"]
        if batch_mode:
            query_embedding = batch_encoding(para)
        local_case = {
            "id": i_case["id"],
            "embedding": query_embedding
        }
        case_list.append(local_case)
    return case_list

def batch_generator(embeddings, batch_size=1000):
    start = 0
    stop = 0
    while stop < embeddings.shape[0] :
        stop = stop + batch_size
        yield embeddings[start:stop]
        start = stop

def calculate_batch_cosine_score(query_list, article_list):
    query_list = np.array(query_list)
    article_list = np.array(article_list)
    query_batch_size=2000
    article_batch_size=500
    output_list = []
    for i, query_emb in enumerate(batch_generator(query_list, batch_size=query_batch_size)):
        for j, article_emb in enumerate(batch_generator(article_list, batch_size=article_batch_size)):
            a = tf.constant (query_emb)
            b = tf.constant (article_emb)
            similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
            similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
            output_list.append(similarity.numpy())
    return output_list

def get_score_list(query_list, embedding_list):
    article_list = []
    id_list = []
    for i_embedding in embedding_list:
        local_id = i_embedding["id"]
        embeddings = i_embedding["embedding"]
        for i_e in embeddings:
            local_i_e = np.array(i_e).reshape(1, -1)
            local_i_e = local_i_e.astype('float32')
            article_list.append(local_i_e)
            id_list.append(local_id)
            
    querys = []
    for i in tqdm(range(len(query_list))):
        data_case = query_list[i]
        query = model_encoding(data_case["text"])
        local_query = query.reshape(1, -1)
        local_query = local_query.astype('float32')
        querys.append(local_query)
    list_consine_score = calculate_batch_cosine_score(querys, article_list)
    finnal_consine_score = list_consine_score[0]
    for i in range(1, len(list_consine_score)):
        finnal_consine_score = np.concatenate((finnal_consine_score, list_consine_score[i]), axis=1)
    return finnal_consine_score, id_list

def mapping_score(query_list, finnal_consine_score, id_list):
    score_data = []
    for i in tqdm(range(len(query_list))):
        local_id_list = id_list
        score_list = []
        for i_article in finnal_consine_score[i]:
            score_list.append(i_article[0])
        old_id = "-1"
        global_max_score = []
        max_score = 0
        for j in range(len(local_id_list)):
            local_id = local_id_list[j]
            if old_id != local_id or j == len(local_id_list) - 1:
                global_max_score.append({
                    "law_serial_number": old_id,
                    "cosine_score": max_score,
                })
                old_id = local_id_list[j]
                max_score = score_list[j]
            elif old_id == local_id:
                max_score = max(max_score, score_list[j])
        global_max_score = sorted(global_max_score, key=lambda d: d['cosine_score'], reverse = True)
        local_case_case = {
            "question_id": query_list[i]["question_id"],
            "text": query_list[i]["text"],
            "relevant_articles": query_list[i]["relevant_articles"],
            "score": global_max_score[:500]
        }
        score_data.append(local_case_case)
    return score_data

def run_cal_cosine_score(data_list, embedding_list):
    finnal_consine_score, id_list = get_score_list(data_list, embedding_list)
    consine_score_list = mapping_score(data_list, finnal_consine_score, id_list)
    return consine_score_list

def run_cal(query_path, law_path):
    query_list = read_json(query_path)
    law_list = read_json(law_path)
    embeding_list = infer_law_list(law_list)
    return run_cal_cosine_score(query_list, embeding_list)

def calculate_recall(predict, label):
    n_true = 0
    for i in label:
        if i in predict:
            n_true += 1
    return n_true

def calculate_recall_limit(case, limit_case):
    marco_recall = 0
    micro_recall = 0
    n_label = 0
    for i in range(len(case)):
        i_case = case[i]
        relevant_laws_list = []
        predict_list = []
        local_n_label = len(i_case["relevant_articles"])
        for i_label in i_case["relevant_articles"]:
            relevant_laws_list.append(i_label["law_serial_number"])
        for i_predict in i_case["score"][:limit_case]:
            predict_list.append(i_predict['law_serial_number'])
        recall = calculate_recall(predict_list, relevant_laws_list)
        n_label += local_n_label
        marco_recall += (recall / local_n_label)
        micro_recall += recall
    micro_recall /= n_label
    marco_recall /= len(case)
    return micro_recall

def run_eval(case):
    eval_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    score_list = []
    for limit_case in eval_list:
        score_list.append(calculate_recall_limit(case, limit_case))
    results = {
        "eval_list": eval_list,
        "score_list": score_list
    }
    return results

def eval_lex_bi(query_path, law_path, out_path):
    query_list = read_json(query_path)
    law_list = read_json(law_path)
    if FAST_DEV_RUN == 1:
        query_list = query_list[:10]
        law_list = law_list[:100]
    embeding_list = infer_law_list(law_list)
    out = []
    limit = 100
    for i in range(0, len(query_list), limit):
        out.extend(run_cal_cosine_score(query_list[i: i + limit], embeding_list))
    outputs = run_eval(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)    

device = DEVICE
model_name = CHECKPOINT
model = SentenceTransformer(model_name, device=device)

if __name__ == "__main__":
    eval_lex_bi(TESTING_PATH, CORPUS_PATH, OUTPUT_DIR + "eval.json")
    print("Done")