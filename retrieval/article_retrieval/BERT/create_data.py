import json
import random
import pandas as pd

from tqdm import tqdm

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def query_articles(case, corpus):
    # Query articles from database
    id_Law = case["law_serial_number"]
    for i_corpus in corpus:
        if i_corpus["law_id"] == id_Law:
            scope_list = [i_corpus["law_name"], case["article_name"]]
            scope_list.extend(i_corpus["level"]["scope"])
            return scope_list


def get_random_case(ran_len, max_ran_len, have_id_list, corpus):
    random_list = []
    while len(random_list) < ran_len:
        random_num = random.randint(0, max_ran_len - 1)
        local_id = corpus[random_num]["id"]
        if local_id not in have_id_list:
            random_list.append(corpus[random_num])
            have_id_list.append(random_num)

    random_sentence_list = []
    for i_case in random_list:
        random_sentence_list.append(i_case["law_name"].lower())
        random_sentence_list.extend(i_case['level']["scope"])
    random_sentence_list = list(set(random_sentence_list))
    return random_sentence_list


def get_full_time(case, corpus_data):
    law_list = case["relevant_articles"]
    try:
        sentence_list = []
        for i_relevent_article in law_list:
            # print(i_relevent_article)
            sentence_list += query_articles(i_relevent_article, corpus_data)
        final_sentence = []
        for i_sentence in sentence_list:
            if len(i_sentence) > 15:
                i_sentence = i_sentence.replace("\n", " ")
                i_sentence = i_sentence.replace("\t", " ")
                i_sentence = i_sentence.replace("\r", " ")
                i_sentence = " ".join(i_sentence.split())
                final_sentence.append(i_sentence.lower())

        return list(set(final_sentence))
    except:
        return []


def process_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text


def get_negative_postive_case(case, corpus, ran_len=5):
    query_list, article_list, label_list = [], [], []
    id_law_list = []
    for i_relevent_article in case["relevant_articles"]:
        id_law_list.append(i_relevent_article["law_serial_number"])
    # print(id_law_list)
    # Get positive case
    positive_case = get_full_time(case, corpus)
    # print(positive_case)
    # Get negative case
    negative_case = get_random_case(ran_len, len(corpus), id_law_list, corpus)
    query_case = case["text"]
    query_case = query_case.lower()
    for i_pos in positive_case:
        query_list.append(process_text(query_case))
        article_list.append(process_text(i_pos))
        label_list.append(1)
    for i_neg in negative_case:
        query_list.append(process_text(query_case))
        article_list.append(process_text(i_neg))
        label_list.append(0)
    return {"query": query_list, "article": article_list, "label": label_list}


def create_data(train_path, corpus_path, ran_len=5):
    train_data = read_json(train_path)
    corpus_data = read_json(corpus_path)
    query_list, article_list, label_list = [], [], []
    for i in tqdm(range(len(train_data))):
        i_case = train_data[i]
        result = get_negative_postive_case(i_case, corpus_data, ran_len)
        query_list += result["query"]
        article_list += result["article"]
        label_list += result["label"]
    df = pd.DataFrame(
        {"query": query_list, "article": article_list, "label": label_list}
    )
    return df