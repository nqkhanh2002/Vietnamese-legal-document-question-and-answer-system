import pandas as pd
import json
import os
import random

def read_json(input_path):
    """
    Read json file from input path
    :param input_path: path to json file
    :return: list of json object
    """
    # print(input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(output_path, data):
    """
    Write json file to output path
    :param output_path: path to json file
    :param data: list of json object
    :return: 
    """
    file_exists = os.path.exists(output_path)

    if not file_exists:
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # print("File created: " + output_path)
    else:
        print("File already exists!")

def write_csv(output_path, data):
    """
    Write csv file to output path
    :param output_path: path to csv file
    :param data: list of json object
    :return: 
    """
    file_exists = os.path.exists(output_path)

    if not file_exists:
        data.to_csv(output_path, index=False)
        # print("File created: " + output_path)
    else:
        print("File already exists!")

def get_list(input_path):
    list_path = os.listdir(input_path)
    for i in range(len(list_path)):
        list_path[i] = os.path.join(input_path, list_path[i])
    return list_path

def random_num(end_ran):
        return random.randint(0, end_ran-1)

def concat_csv(csv_list):
    df = pd.concat(csv_list, ignore_index=True)
    return df

# print(get_list("output/"))