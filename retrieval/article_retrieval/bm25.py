import json
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from retrieval.support_func import *

def min_max_scale(data):
    local_max = max(data)
    local_min = min(data)
    output_list = []
    for local_data in data:
        try:
            output_list.append((local_data - local_min)/(local_max - local_min))
        except:
            output_list.append(-1)
        
    return output_list

def pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())

def get_local_content_list(corpus):
    content_list = []
    
    for data in corpus:
        relevant_article = data['paragraph']
        for article in relevant_article:
            local_id = "[" + str(data['db_id']) + "]"
            local_article_text = pre_processing(article)
            content_list.append(local_id + local_article_text)

    return content_list

def create_bm_corpus(corpus):
    content_list = get_local_content_list(corpus)

    print("BM25: Creating corpus...")   
            
    tokenized_corpus = [doc.split(" ") for doc in content_list]

    print("BM25: Creating model...")
    
    return BM25Okapi(tokenized_corpus), content_list

def answer_split(answer):
    answer = answer.split("]")
    text = answer[1]
    db_id = answer[0][1:]
    return text, db_id

def single_query_bm25(case, corpus, bm25_model, topk):
    query = pre_processing(case['text'])
    tokenized_query = query.split(" ")
    scores = bm25_model.get_scores(tokenized_query)
    score_list = min_max_scale(scores)
    top_list = np.argsort(score_list)[::-1][:topk]
    output_list = []
    for i in top_list:
        text, law_id = answer_split(corpus[i])
        local_data = {
            "law_serial_number": law_id,
            "score": score_list[i]
        }
        output_list.append(local_data)
    case['predict_articles'] = output_list
    return case

def multi_query_bm25(query_list, corpus, bm25_model, topk):
    output_list = []
    for i in tqdm(range(len(query_list))):
        query = query_list[i]
        output_list.append(single_query_bm25(query, corpus, bm25_model, topk))
    return output_list

def run_bm25(corpus_path, dataset_path, topk):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    bm25_model, content_list = create_bm_corpus(corpus)

    output_list = multi_query_bm25(dataset, content_list, bm25_model, topk)
    return output_list

def eval_bm25(bm25_path = 'resource/bi_bm25_output.json', limit = 5):
    with open(bm25_path, 'r', encoding='utf-8') as f:
        bm25_list = json.load(f)

    output_list = []
    number_label = 0
    number_true = 0
    for data in bm25_list:
        predict_list = data['predict_articles']
        label_list = data['relevant_articles']
        number_label += len(label_list)
        predict_id = [str(x['law_serial_number']) for x in predict_list]
        label_id = [str(x['id']) for x in label_list]
        
        predict_id = predict_id[:limit]
        for i in range(len(predict_id)):
            if predict_id[i] in label_id:
                number_true += 1
                break

    return {
        "topk": limit,
        "number_label": number_label,
        "number_true": number_true,
        "recall": number_true/number_label
    }
