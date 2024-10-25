import json
import random
import os
import json
import re
import numpy as np
import pandas as pd
import random
import math

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from rank_bm25 import BM25Okapi
from retrieval.constant import *

number_of_negative_samples = NEGATIVE_NUM

# Function to find a random article from the same law but different from the correct article
def find_hard_negative(law, relevant_id):
    # Flatten all sections and articles in the law except the correct one
    articles = []
    for chapter in law['content']:
        for section in chapter['content_Chapter']:
            for article in section['content_Section']:
                if (chapter['id_Chapter'], section['id_Section'], article['id_Article']) not in relevant_id:
                    articles.append(article['content_Article'])
    return random.sample(articles, min(number_of_negative_samples, len(articles))) if articles else None

# Function to find a random article from a different law
def find_soft_negative(law_dict, correct_law_id, num_laws=2, num_chapters=2, num_sections=2, num_articles=2):
    random_articles = []

    while len(random_articles) < number_of_negative_samples:
        random_law_ids = random.sample([key for key in law_dict.keys() if key != correct_law_id], num_laws)
        
        for random_law_id in random_law_ids:
            random_law = law_dict[random_law_id]
            random_chapters = random.sample(random_law['content'], min(num_chapters, len(random_law['content'])))
#             print(len(random_chapters))
            for random_chapter in random_chapters:
                random_sections = random.sample(random_chapter['content_Chapter'], min(num_sections, len(random_chapter['content_Chapter'])))
#                 print(len(random_sections))
                for random_section in random_sections:
                    random_articles_in_section = random.sample(random_section['content_Section'], min(num_articles, len(random_section['content_Section'])))
                    
                    for random_article in random_articles_in_section:
                        random_articles.append(random_article['content_Article'])
    
    return random.sample(random_articles, min(number_of_negative_samples, len(random_articles))) if random_articles else None

# Function to process the questions and match the relevant laws
def process_questions(question_data, law_dict):
    save_data = []
    for question in question_data:
        relevant_id = []
        for relevant_law in question['relevant_laws']:
            chapter_id = relevant_law['id_Chapter']
            section_id = relevant_law['id_Section']
            article_id = relevant_law['id_Article']
            relevant_id.append((chapter_id, section_id, article_id))
            
        for relevant_law in question['relevant_laws']:
            law_id = relevant_law['id_Law']
            chapter_id = relevant_law['id_Chapter']
            section_id = relevant_law['id_Section']
            article_id = relevant_law['id_Article']

            # Find the corresponding law
            if law_id in law_dict:
                law = law_dict[law_id]
                for chapter in law['content']:
                    if chapter['id_Chapter'] == chapter_id:
                        for section in chapter['content_Chapter']:
                            if section['id_Section'] == section_id:
                                for article in section['content_Section']:
                                    if article['id_Article'] == article_id:
                                        # Add the content of the article to the relevant law in the question
                                        relevant_law['content'] = article['content_Article']

                                        # Add a soft negative
                                        relevant_law['soft_negative'] = find_soft_negative(law_dict, law_id)

                                        # Add a hard negative
                                        relevant_law['hard_negative'] = find_hard_negative(law, relevant_id)
            
        save_data.append(question)
    return save_data

def random_num(start_ran, end_ran):
    return random.randint(start_ran, end_ran)

def load_json(path):
    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    return data

def query_articles(query, corpus):
    query_law_id = query["law_id"]
    query_article_id = query["article_id"]

    for local_law in corpus:
        local_id = local_law["id"]
        if local_id == query_law_id:
            for local_article in local_law["articles"]:
                local_article_id = local_article["id"]
                if local_article_id == query_article_id:
                    return "[Điều " + query_article_id + " "+ query_law_id + "] " + local_article["text"]
            
    return None

def rand_articles(query, corpus):
    query_law_id = query["law_id"]
    query_article_id = query["article_id"]

    for local_law in corpus:
        local_id = local_law["id"]
        if local_id == query_law_id:
            limit_len = len(local_law["articles"])
            query_article_id = str(random_num(limit_len))
            for local_article in local_law["articles"]:
                local_article_id = local_article["id"]
                if local_article_id == query_article_id:
                    return "[Điều " + query_article_id + " "+ query_law_id + "] " + local_article["text"]
    
    return None

def rand_database(list_rand, avoid_rule, num):
    cnt = 0
    output_list = []
    while cnt < num:
        ran_idx = random_num(50)
        try:
            local_aricle = list_rand[ran_idx]
            local_detect = detect_law(local_aricle)
            if local_detect == avoid_rule: continue
        except:
            continue
        output_list.append(local_aricle["text"])
        cnt += 1
    return output_list

def concat_data(dataset_path, corpus_path):
    local_dataset = load_json(dataset_path)
    local_corpus = load_json(corpus_path)

    query_list = []
    answer_list = []
    label_list = []
    for local_data in local_dataset:
        local_id = local_data["question_id"]
        local_text = local_data["text"]
        try:
            local_choices = local_data["choices"]
            local_choices = local_choices["A"] + ". " + local_choices["B"] + ". " + local_choices["C"] + ". " + local_choices["D"]
            local_text += " " + local_choices
        except:
            local_text = local_text
        
        local_articles = query_articles(local_data["relevant_articles"][0], local_corpus)
        query_list.append(pre_processing(local_text))
        answer_list.append(pre_processing(local_articles))
        label_list.append(1)

        
    return {
        "query": query_list, 
        "article": answer_list, 
        "label": label_list
    }


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())

def write_jsonfile(filename, data):
    result = json.dumps(data, indent=4, ensure_ascii=False)
    myjsonfile = open(filename, "w", encoding="utf8")
    myjsonfile.write(result)
    myjsonfile.close()
    
def create_bm_corpus(corpus_path):
    corpus = load_json(corpus_path)
    output_corpus_list = []
    for local_law in corpus:
        local_id = local_law["id"]
        for lo in local_law["articles"]:
            content = "[Điều " + lo["id"] + " "+ local_id + "] " + pre_processing(lo["text"])
            output_corpus_list.append(content)
            
    tokenized_corpus = [doc.split(" ") for doc in output_corpus_list]
    
    return BM25Okapi(tokenized_corpus), output_corpus_list

def minmax_scale(score_list, eps = 0.0001):
    score_list = np.array(score_list)
    min_score = np.min(score_list)
    max_score = np.max(score_list)
    if max_score - min_score == 0:
        return [1.0] * len(score_list)
    return (score_list - min_score) / (max_score - min_score + eps)

def query_bm25(query, corpus, bm25_model, n):
    query = pre_processing(query)
    tokenized_query = query.split(" ")
    score_list = bm25_model.get_scores(tokenized_query)
    score_list = minmax_scale(score_list)
    top_list = np.argsort(score_list)[::-1][:n]
    output_list = []
    for i in top_list:
        local_data = {
            "score": score_list[i],
            "text": corpus[i]
        }
        output_list.append(local_data)
    return output_list
            
def get_top_n_list(dataset_path, corpus_path, top_k=50):
    local_dataset = load_json(dataset_path)
    bm25, corpus = create_bm_corpus(corpus_path)
    
    output_list = []
    
    for local_data in local_dataset:
        local_id = local_data["question_id"]
        local_text = local_data["text"]
        try:
            local_choices = local_data["choices"]
            local_choices = local_choices["A"] + ". " + local_choices["B"] + ". " + local_choices["C"] + ". " + local_choices["D"]
            local_text += " " + local_choices
        except:
            local_text = local_text
        local_data = {
            "question_id": local_id,
            "relevant_articles": local_data["relevant_articles"],
            "text": local_text,
            "top_n": query_bm25(local_text, corpus, bm25, top_k)
        }
        output_list.append(local_data)
        
    return output_list

def concat_file(dataset_path, corpus_path, is_train):
    local_dataset = load_json(dataset_path)
    local_corpus = load_json(corpus_path)
        
    query_list = []
    answer_list = []
    label_list = []
    for local_data in local_dataset:
        local_id = local_data["question_id"]
        local_text = local_data["text"]
        if is_train:
            local_articles = query_articles(local_data["relevant_articles"][0], local_corpus)
            query_list.append(pre_processing(local_text))
            answer_list.append(pre_processing(local_articles))
            label_list.append(1)
        for candidate in local_data["top_n"]:
            query_list.append(local_text)
            answer_list.append(candidate["text"])
            label_list.append(0)
        
    return {
        "query": query_list, 
        "article": answer_list, 
        "label": label_list
    }

def detect_law(data):
    score = data["score"]
    data = data["text"]
    content = "".join(data.split("]")[1:])
    data = (data.split("]")[0])[1:]
    data = data.split()
    return {
        "law_id": " ".join(data[2:]),
        "article_id": " ".join(data[1:2]),
        "score": score,
        "content": content
    }

def create_output(data,top_k=1):
    result_list = []
    for local_data in data:
        top_n_items = local_data["top_n"][:top_k]
        relevant_articles = [detect_law(item) for item in top_n_items]
        data_output = {
            "question_id": local_data["question_id"],
            "question_content": local_data["text"],
            "relevant_articles": local_data["relevant_articles"],
            "bm25_relevant_articles": relevant_articles
        }
        result_list.append(data_output)
    return result_list

def check_in_label(article, relevant_articles_list):
    law_id = article["law_id"]
    article_id = article["article_id"]
    for relevant_articles in relevant_articles_list:
        if law_id == relevant_articles["law_id"] and article_id == relevant_articles["article_id"]:
            return 1
    return 0

def make_df(data, rate = 1):
    question_list = []
    article_list = []
    bm25_score_list = []
    label_list = []
    for i in range(len(data)):
        cnt_neg = 0
        cnt_pos = 0
        local_question = data[i]["question_content"]
        for j in range(len(data[i]["bm25_relevant_articles"])):
            local_article = data[i]["bm25_relevant_articles"][j]["content"]
            local_label = int(check_in_label(data[i]["bm25_relevant_articles"][j], data[i]["relevant_articles"]))
            local_bm25_score = data[i]["bm25_relevant_articles"][j]["score"]
            if local_label == 1:
                question_list.append(local_question)
                article_list.append(local_article)
                bm25_score_list.append(local_bm25_score)
                label_list.append(local_label)
                cnt_pos += 1
        
        for j in range(len(data[i]["bm25_relevant_articles"])):
            local_article = data[i]["bm25_relevant_articles"][j]["content"]
            local_label = int(check_in_label(data[i]["bm25_relevant_articles"][j], data[i]["relevant_articles"]))
            local_bm25_score = data[i]["bm25_relevant_articles"][j]["score"]
            if local_label == 0:
                question_list.append(local_question)
                article_list.append(local_article)
                bm25_score_list.append(local_bm25_score)
                label_list.append(local_label)
                cnt_neg += 1
                if cnt_neg == rate * cnt_pos:
                    break

    case = {
        "question_list": question_list,
        "article_list": article_list,
        "bm25_score_list": bm25_score_list,
        "label_list": label_list,
    }
    return pd.DataFrame(case)

def create_train_alqac_data(dataset_path, corpus_path, output_path, top_k=50, neg_rate=1):
    train_one = get_top_n_list(dataset_path, corpus_path, top_k)
    output_p2 = create_output(train_one, top_k)
    output_p3 = make_df(output_p2, neg_rate)
    output_p3.to_csv(output_path, index=False)