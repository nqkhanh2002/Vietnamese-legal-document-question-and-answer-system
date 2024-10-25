import json
import os

from tqdm import tqdm
from retrieval.constant import *

def concat_json(json_list_path):
    json_concated_list = []
    for json_path in json_list_path:
        with open(json_path, "r", encoding="utf-8") as f:
            json_list = json.load(f)
            json_concated_list += json_list

    return json_concated_list


def concat_json_article(json_list_path):
    json_concated_list = []
    for json_path in json_list_path:
        with open(json_path, "r", encoding="utf-8") as f:
            json_list = json.load(f)
            # print(json_list)
            json_list.pop("predict_relevant_article")
            json_concated_list.append(json_list)
            # break
    # print(json_concated_list)
    return json_concated_list


def run_concat(input_path, output_path):
    json_concated_list = concat_json(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_concated_list, f, indent=2, ensure_ascii=False)

    print(f"Concatenated json files are saved at {output_path}")


def run_json_concat(input_path, output_path):
    json_concated_list = concat_json_article(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_concated_list, f, indent=2, ensure_ascii=False)

    print(f"Concatenated json files are saved at {output_path}")


def concat_law(source_path, additional_path, output_path):
    with open(source_path, "r", encoding="utf-8") as f:
        source_list = json.load(f)
    with open(additional_path, "r", encoding="utf-8") as f:
        additional_list = json.load(f)

    have_case = 0

    new_list = []

    for i in range(len(additional_list)):
        add_law_id = additional_list[i]["id"]
        for j in range(len(source_list)):
            if add_law_id == source_list[j]["law_id"]:
                have_case += 1
                # source_list[j]["predict_relevant_article"] = additional_list[i]["predict_relevant_article"]
                additional_list[i]["law_name"] = source_list[j]["law_name"]
                additional_list[i]["title"] = ". ".join(
                    source_list[j]["level"]["title"]
                )
                additional_list[i]["scope"] = ". ".join(
                    source_list[j]["level"]["scope"]
                )
                new_list.append(additional_list[i])
                break
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_list, f, indent=2, ensure_ascii=False)

    return {
        "source_list": len(source_list),
        "additional_list": len(additional_list),
        "have_case": have_case,
    }


def create_infer_40k_list(data_path, output_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    infer_list = []
    for data in data_list:
        local_id = data["law_id"]
        db_id = data["id"]
        infer_case = []
        infer_case.append(". ".join(data["level"]["title"]))
        infer_case.append(". ".join(data["level"]["scope"]))
        infer_case += data["level"]["article_title_list"]

        for i in range(len(infer_case)):
            infer_case[i] = infer_case[i].lower()
            infer_case[i] = infer_case[i].replace("-", "")
            infer_case[i] = infer_case[i].replace("\n", "")
            infer_case[i] = infer_case[i].replace("\t", "")
            infer_case[i] = " ".join(infer_case[i].split())
        infer_case = list(set(infer_case))
        local_case = {"id": local_id, "db_id": db_id, "paragraph": infer_case}
        infer_list.append(local_case)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(infer_list, f, indent=2, ensure_ascii=False)

    return infer_list


def create_infer_list(data_path, output_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    infer_list = []
    for data in data_list:
        local_id = data["id"]
        infer_case = []
        try:
            if data["law_name"] == data["scope"]:
                infer_case.append(data["scope"])
            else:
                infer_case.append(data["law_name"])
                infer_case.append(data["scope"])
            for i_Chaptert in data["content"]:
                if len(i_Chaptert["title_Chapter"]) > 15:
                    infer_case.append(i_Chaptert["title_Chapter"])
                for i_Section in i_Chaptert["content_Chapter"]:
                    if len(i_Section["title_Section"]) > 15:
                        infer_case.append(i_Section["title_Section"])
                    for i_Article in i_Section["content_Section"]:
                        if len(i_Article["title_Article"]) > 15:
                            infer_case.append(i_Article["title_Article"])

            for i in range(len(infer_case)):
                infer_case[i] = infer_case[i].lower()

            local_case = {"id": local_id, "paragraph": infer_case}
            infer_list.append(local_case)
        except:
            pass

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(infer_list, f, indent=2, ensure_ascii=False)

    return infer_list


def mapping_law_article(meta_data_path, article_path, output_path):
    with open(meta_data_path, "r", encoding="utf-8") as f:
        meta_data_list = json.load(f)
    with open(article_path, "r", encoding="utf-8") as f:
        article_list = json.load(f)
    meta_data_case = meta_data_list.copy()
    # for i in range(len(meta_data_list)):
    #     meta_data_case += meta_data_list[i]
    # print(meta_data_case)
    for i in tqdm(range(len(meta_data_case))):
        meta_data_id = meta_data_case[i]["id"]
        # print(meta_data_id)
        for j in range(len(article_list)):
            if str(meta_data_id) == str(article_list[j]["vbpl_id"]):
                # article_list[j]['section_number'] = int(article_list[j]["section_number"])
                meta_data_case[i]["level"]["article_list"].append(article_list[j])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_data_case, f, indent=2, ensure_ascii=False)

