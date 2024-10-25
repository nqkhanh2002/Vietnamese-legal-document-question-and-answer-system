import src.support_func as support_func
import pandas as pd
import os
from tqdm import tqdm
# '''
# "content": 
#     "chapter": list
#         "section": list
#             "article": list
# '''

# demo_link = "dataset\corpus\law_3.json"

def process_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    if len(text) <= 1:
        return " "
    return text

def get_article_content(law_id, article_id, corpus):
    for local_law in corpus:
        local_law_id = local_law["id"]
        if law_id != local_law_id:
            continue
        for chapter in local_law["content"]:
            for section in chapter["content_Chapter"]:
                for article in section["content_Section"]:
                    local_article_id = article["id_Article"]
                    if str(article_id) == str(local_article_id):
                        return article["content_Article"]
    return None

def get_article_list(corpus):
    corpus_id = corpus["id"]
    article_list = []
    for chapter in corpus["content"]:
        for section in chapter["content_Chapter"]:
            for article in section["content_Section"]:
                id_article = article["id_Article"]
                local_content = article["content_Article"]
                local_article = "[" + str(id_article) + "-" + str(corpus_id) + "] " + local_content
                article_list.append(local_article)
    return article_list

def get_article_list_lex(corpus):
    article_list = []
    corpus_id = corpus["law_id"]
    for article in corpus["level"]["article_list"]:
        try:
            local_article = "[" + str(article["section_number"]) + "-" + str(corpus_id) + "-" + str(article["id"]) + "@#$" + str(article["vbpl_id"]) + "] " + article["section_content"]
            article_list.append(local_article)
        except:
            pass
    return article_list


def get_content_list(corpus):
    content_list = []
    
    for data in corpus:
        content_list += get_article_list_lex(data)

    return content_list

# kill all
def get_none_list(qa_corpus):
    not_none_list = []
    none_list = []
    flag = 0
    for data in qa_corpus:
        flag = 0
        for case in data["relevant_laws"]:
            if case["id_Article"] == "None":
                none_list.append(data)
                flag = 1
                break
        if flag == 0:
            not_none_list.append(data)
    return none_list, not_none_list

# demo_qa = "dataset\question\question_9.1.json"
# qa_list = support_func.read_json(demo_qa)
# none_list, not_none_list = get_none_list(qa_list)
# support_func.write_json("none_1.json", none_list)
# support_func.write_json("not_none_1.json", not_none_list)

def equal_case(case_1, case_2):
    if case_1["id_Law"] != case_2["id_Law"]:
        return False
    if case_1["id_Article"] != case_2["id_Article"]:
        return False
    return True

def equal_case_list(case, list_case):
    if len(list_case) == 0:
        return False
    for data in list_case:
        if equal_case(case, data):
            return True
    return False

def remove_same_label(qa_case):
    label_list = qa_case["relevant_laws"]
    new_label_list = []
    for case in label_list:
        if equal_case_list(case, new_label_list):
            continue
        new_label_list.append(case)
    qa_case["relevant_laws"] = new_label_list
    return qa_case

def create_csv_format(case, data_list, label_type):
    id_list = []
    question_list = []
    id_law_list = []
    id_article_list = []
    # content_list = []
    label_list = []

    for data in data_list:
        id_list.append(case["id"])
        question_list.append(case["question"])
        id_law_list.append(data["id_Law"])
        id_article_list.append(data["id_Article"])
        # content_list.append(data["text"])
        label_list.append(label_type)

    column_name = ["id", "question", "id_Law", "id_Article", "label"]
    concat_list = list(zip(id_list, question_list, id_law_list, id_article_list, label_list))
    df = pd.DataFrame(concat_list, columns = column_name)
    return df

def create_negative(predict_list, label_list, ran_num):
    label_0_law_list = []
    while len(label_0_law_list) < ran_num:
        rand = support_func.random_num(len(predict_list))
        
        local_case = predict_list[rand]
        if equal_case_list(local_case, label_list):
            continue
        label_0_law_list.append(local_case)
    return label_0_law_list

def create_single(case, ran_num):
    predict_list = case["predict_relevant_article"]
    label_list = case["relevant_laws"]
    # print(ran_num)
    ran_num *= len(label_list)
    # print(ran_num)
    label_0_law_list = create_negative(predict_list, label_list, ran_num)

    label_0_csv = create_csv_format(case, label_0_law_list, 0)
    label_1_csv = create_csv_format(case, label_list, 1)

    return support_func.concat_csv([label_0_csv, label_1_csv])

def get_law_content(law_serial_number, article_number, corpus):
    # print(law_serial_number, article_number)
    for law in corpus:
        if law["law_id"] != law_serial_number:
            continue
        # print("law_id: ", law["law_id"])
        for aritcle in law["level"]["article_list"]:
            try:
                if str(aritcle["section_number"]) == str(article_number):
                    # print("article_number: ", aritcle["section_number"])
                    content = aritcle["section_content"]
                    # print(content)
                    if content is None:
                        content = "-1"
                    return content
            except:
                pass
    return "-1"

def create_multi(case_list, ran_num):
    output_list = []
    for case in case_list:
        case = support_func.read_json(case)
        local_df = create_single(case, ran_num)
        output_list.append(local_df)
    return support_func.concat_csv(output_list)

def check_not_in_label(case, label_list):
    for data in label_list:
        if data["law_serial_number"] == case["law_serial_number"] and data["article_number"] == case["article_number"]:
            try:
                return data["score"]
            except:
                return 1
    return 0.0001

def create_single_case(case, corpus, mode, num_negative):
    case = support_func.read_json(case)
    relevant_articles_list = case["relevant_articles"]
    predict_relevant_article_list = case["predict_relevant_article"]
    negative_list = []
    i_predict = 0
    num_negative = min(num_negative * len(relevant_articles_list), len(predict_relevant_article_list))

    for i in range(len(relevant_articles_list)):
        case["relevant_articles"][i]["bm25_score"] = check_not_in_label(relevant_articles_list[i], predict_relevant_article_list)
        if len(case["relevant_articles"][i]["article_content"].split()) < 2:
            return [], [], [], []

    while len(negative_list) < num_negative:
        if mode == "soft":
            rand = support_func.random_num(len(predict_relevant_article_list))
            local_case = predict_relevant_article_list[rand]
            if check_not_in_label(local_case, negative_list):
                negative_list.append(local_case)
        elif mode == "hard":
            if i_predict >= len(predict_relevant_article_list):
                break
            local_case = predict_relevant_article_list[i_predict]
            if check_not_in_label(local_case, relevant_articles_list) == 0.0001:
                local_case["article_content"] = get_law_content(local_case["law_serial_number"], local_case["article_number"], corpus)
                if local_case["article_content"] == "-1":
                    i_predict += 1
                    continue
                negative_list.append(local_case)
            # print(i_predict, check_not_in_label(local_case, relevant_articles_list))
            i_predict += 1
        elif mode == "mix":
            rand = support_func.random_num(len(relevant_articles_list))
            local_case = relevant_articles_list[rand]
            if check_not_in_label(local_case, negative_list):
                negative_list.append(local_case)
            if i_predict >= len(predict_relevant_article_list):
                break
            local_case = relevant_articles_list[i_predict]
            if check_not_in_label(local_case, negative_list):
                local_case["article_content"] = get_law_content(local_case["law_serial_number"], local_case["article_number"], corpus)
                negative_list.append(local_case)
            i_predict += 1

    question_list = []
    article_list = []
    label_list = []
    bm25_list = []

    for data in relevant_articles_list:
        question_list.append(process_text(case["text"]))
        article_list.append(process_text(data["article_content"]))
        bm25_list.append(data["bm25_score"])
        label_list.append(1)

    for data in negative_list:
        question_list.append(process_text(case["text"]))
        article_list.append(process_text(data["article_content"]))
        bm25_list.append(data["score"])
        label_list.append(0)

    return question_list, article_list, bm25_list, label_list

def create_multi_case(case_folder_path, corpus_path, mode, num_negative, train_test_flag = "train"):
    corpus = support_func.read_json(corpus_path)
    case_list = [os.path.join(case_folder_path, file) for file in os.listdir(case_folder_path)]
    question_list = []
    article_list = []
    label_list = []
    bm25_list = []

    # if train_model == "train":
    #     for i in tqdm(range(len(case_list))):
    #         case = case_list[i]
    #         question, article, label = create_single_case(case, corpus, mode, num_negative)
    #         question_list += question
    #         article_list += article
    #         label_list += label
        
    #     column_name = ["question", "article", "label"]
    #     concat_list = list(zip(question_list, article_list, label_list))
    #     df = pd.DataFrame(concat_list, columns = column_name)
    # elif train_model == "test":
    for i in tqdm(range(len(case_list))):
        case = case_list[i]
        question, article, bm25, label = create_single_case(case, corpus, mode, num_negative)
        if len(question) == 0 and len(article) == 0 and len(label) == 0 and len(bm25) == 0:
            continue
        question_list += question
        article_list += article
        label_list += label
        bm25_list += bm25

    column_name = ["question", "article", "label", "bm25_score"]
    concat_list = list(zip(question_list, article_list, label_list, bm25_list))
    df = pd.DataFrame(concat_list, columns = column_name)
    if train_test_flag == 'train':
        df = df.sample(frac = 1).reset_index(drop = True)
    return df

if __name__ == "__main__":
    corpus_path = "resource/retrain/Lex_T9/T9_data/mapped_law.json"

    case = "resource/retrain/Lex_T9/T9_data/Testset"
    create_multi_case(case, corpus_path, "hard", 5, train_test_flag='test').to_csv("00.csv", index = False)