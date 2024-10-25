from rank_bm25 import BM25Okapi
import numpy as np
from src import support_func
from src.train_cross_encoder import create_data
import argparse
import os

from tqdm import tqdm
# from retrieval.constant import *

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
        relevant_article = data['list_relevant_article']
        for article in relevant_article:
            local_id = "[" + str(article['article_id']) + "-" + str(article['law_id']) + "]"
            local_article_text = pre_processing(article['article_text'])
            content_list.append(local_id + local_article_text)
        
    return content_list

def create_bm_corpus(corpus):

    content_list = create_data.get_content_list(corpus)

    print("BM25: Creating corpus...")   
            
    tokenized_corpus = [doc.split(" ") for doc in content_list]

    print("BM25: Creating model...")
    
    return BM25Okapi(tokenized_corpus), content_list

def answer_split(answer):
    answer = answer.split("]")
    text = answer[1]
    spl_ans = answer[0].split("-") 
    article_id = spl_ans[0][1:]
    db_id = spl_ans[-1]
    law_id = "-".join(spl_ans[1:-1])
    article_db_id = db_id.split("@#$")[1]
    law_db_id = db_id.split("@#$")[0]
    return text, law_id, article_id, law_db_id, article_db_id  

def single_query_bm25(query, corpus, bm25_model, topk):
    query = pre_processing(query)
    tokenized_query = query.split(" ")
    score_list = bm25_model.get_scores(tokenized_query)
    score_list = min_max_scale(score_list)
    top_list = np.argsort(score_list)[::-1][:topk]
    output_list = []
    for i in top_list:
        text, law_id, article_id, law_db_id, article_db_id = answer_split(corpus[i])
        local_data = {
            "law_db_id": law_db_id,
            "article_db_id": article_db_id,
            "law_serial_number": law_id,
            "article_number": article_id,
            "score": score_list[i]
        }
        output_list.append(local_data)
    return output_list

def multi_query_bm25(query_list, corpus, output_path, bm25_model, topk, mode, start_idx, end_idx):
    len_query_list = len(query_list)
    cnt = 0
    output_list = []
    # for query in query_list:
    #     cnt += 1
    #     if cnt % 100 == 0:
    #         print("BM25: " + str(cnt) + "/" + str(len_query_list) + " done")
    #     query = pre_processing(query["question"])
    #     if query == "":
    #         output_list.append([])
    #         continue
    #     local_output_list = single_query_bm25(query, corpus, bm25_model, topk)
    #     output_list.append(local_output_list)

    if start_idx == -1:
        start_idx = 0

    if end_idx == -1:
        end_idx = len(query_list)

    for i in tqdm(range(start_idx, end_idx)):
        query = query_list[i]
        cnt += 1
        # if cnt % 100 == 0:
        #     print("BM25: " + str(cnt) + "/" + str(len_query_list) + " done")
        if mode == "single":
            # file_name = output_path + str(query_list[i]["question_id"]) + ".json"
            file_name = os.path.join(output_path, str(query_list[i]["question_id"]) + ".json")
            if os.path.exists(file_name):
                continue        

        query = pre_processing(query["text"])
        if query == "":
            output_list.append([])
            continue
        local_output_list = single_query_bm25(query, corpus, bm25_model, topk)
        query_list[i]["predict_relevant_article"] = local_output_list
        local_query = query_list[i]
        if mode == "multi":
            output_list.append(local_query)
        elif mode == "single":
            file_name = os.path.join(output_path, str(query_list[i]["question_id"]) + ".json")
            support_func.write_json(file_name, local_query)

    return output_list

# mode = "single/multi"
def run_bm25(dataset_path, corpus_path, output_path, topk, mode, start_idx=-1, end_idx=-1):
    dataset = support_func.read_json(dataset_path)
    # dataset = dataset_path
    corpus_set = support_func.read_json(corpus_path)
    bm25_model, corpus = create_bm_corpus(corpus_set)
    print("BM25: Running...")
    query_list = multi_query_bm25(dataset, corpus, output_path, bm25_model, topk, mode, start_idx, end_idx)
    # for i in range(len(query_list)):
    #     local_query = {
    #         "id": dataset[i]["id"],
    #         "href": dataset[i]["href"],
    #         "question": dataset[i]["question"],
    #         "answer": dataset[i]["answer"],
    #         "short_answer": dataset[i]["short_answer"],
    #         "relevant_laws": dataset[i]["relevant_laws"],
    #         "predict_relevant_article": query_list[i]
    #     }
    #     if mode == "multi":
    #         query_list[i] = local_query
    #     elif mode == "single":
    #         support_func.write_json(output_path + str(i) + ".json", local_query)
    if mode == "single":
        return None
    return query_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='resource/qna_9.json', type=str)
    parser.add_argument('--corpus_path', default='resource/mapped_law.json', type=str)
    parser.add_argument('--output_path', default='resource/15_9_Cross/', type=str)
    parser.add_argument('--topk', default=200, type=int)
    parser.add_argument('--mode', default='single', type=str)

    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--end_idx', default=-1, type=int)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    corpus_path = args.corpus_path
    output_path = args.output_path
    topk = args.topk
    mode = args.mode

    start_idx = args.start_idx
    end_idx = args.end_idx

    if mode == "single":
        run_bm25(dataset_path, corpus_path, output_path, topk, mode, start_idx, end_idx)
    elif mode == "multi":
        query_list = run_bm25(dataset_path, corpus_path, output_path, topk, mode)
        support_func.write_json(output_path + "\\bm25_multi.json", query_list)