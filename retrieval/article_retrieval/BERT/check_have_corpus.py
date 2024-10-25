import json

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    print(len(data))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_new_list(query_data, corpus):
    id_law_list = []
    for i in corpus:
        id_law_list.append(i["id"])

    new_list = []
    for i in query_data:
        flag = 0
        for j in i["relevant_laws"]:
            if j["id_Law"] not in id_law_list:
                flag = 1
        if flag == 0:
            new_list.append(i)
    return new_list