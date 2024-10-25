from retrieval.constant import *
from retrieval.support_func import *

import os
import shutil
import random

def get_list_article(data_path, train_folder, test_folder, train_size = 0.8):
    list_long_article = []
    list_short_article = []
    for i_file in os.listdir(data_path):
        local_path = os.path.join(data_path, i_file)
        case = read_json(local_path)
        if len(case["relevant_articles"]) > 2:
            list_long_article.append(i_file)
        else:
            list_short_article.append(i_file)

    # shuffle list
    random.shuffle(list_long_article)
    random.shuffle(list_short_article)

    # split list
    split_index = int(len(os.listdir(data_path)) * (1 - train_size))
    test_list = list_short_article[:split_index]
    train_list = list_short_article[split_index:] + list_long_article

    # save to file
    for i_file in test_list:
        i_file = os.path.join(data_path, i_file)
        shutil.copy(i_file, test_folder)
    
    for i_file in train_list:
        i_file = os.path.join(data_path, i_file)
        shutil.copy(i_file, train_folder)

    print("Train size: ", len(train_list))
    print("Test size: ", len(test_list))

def split_train_test_by_file(train_file, test_file_1, test_file_2, base_folder, train_folder_1, train_folder_2, test_folder_1, test_folder_2):
    train_list = read_json(train_file)
    test_list_1 = read_json(test_file_1)
    test_list_2 = read_json(test_file_2)

    id_train = []
    for i_file in train_list:
        id_train.append(i_file["question_id"])
    
    id_test_1 = []
    for i_file in test_list_1:
        id_test_1.append(i_file["question_id"])

    id_test_2 = []
    for i_file in test_list_2:
        id_test_2.append(i_file["question_id"])
    for i_file in os.listdir(base_folder):
        local_path = os.path.join(base_folder, i_file)
        case = read_json(local_path)
        if case["question_id"] in id_test_1:
            shutil.copy(local_path, test_folder_1)
        elif case["question_id"] in id_test_2:
            shutil.copy(local_path, test_folder_2)
        else:
            if case["question_id"] <= 7000:
                shutil.copy(local_path, train_folder_1)
            else:
                shutil.copy(local_path, train_folder_2)

if __name__ == "__main__":
    # data_path = "D:/Lab/Lex_biencoder/resource/cross_data/bm25"
    # train_folder = "D:/Lab/Lex_biencoder/resource/cross_data/train"
    # test_folder = "D:/Lab/Lex_biencoder/resource/cross_data/test"
    # get_list_article(data_path, train_folder, test_folder, train_size = 0.8)

    split_train_test_by_file('resource/train_infomation.json', 'resource/test_1_infomation.json', 'resource/test_2_infomation.json', 'resource/cross_data', 'resource/train_1', 'resource/train_2', 'resource/test_1', 'resource/test_2')