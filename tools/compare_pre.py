import os

from bs4 import BeautifulSoup
from transformers import AutoConfig, AutoTokenizer

real_suffix = ".TimeNorm.gold.completed.xml"
predict_suffix = ".TimeNorm.system.completed.xml"
mid_dir = os.path.join("time")
real_dir = os.path.join("submission", "ref", mid_dir)
predict_dir = os.path.join("submission", "res", mid_dir)
content_dir = os.path.join("practice_text", mid_dir)


def get_span_type_text(p_file_path):
    with open(p_file_path, "r") as f:  # 打开文件
        p_data = f.read()  # 读取文件
    p_soup = BeautifulSoup(p_data, 'lxml')
    p_entity_list = p_soup.find_all("entity")
    p_entity_text_list = []
    for one_entity in p_entity_list:
        p_entity_text_list.append((one_entity.span.string, one_entity.type.string))
    p_entity_text_list.sort()
    return p_entity_text_list


def get_words_by_idx(p_s_idx, p_data):
    p_idx_words_list = []
    for p_a_idx in p_s_idx:
        p_begin_end_idx = p_a_idx[0].strip("''").split(",")
        p_begin_idx = int(p_begin_end_idx[0])
        p_end_idx = int(p_begin_end_idx[1])
        p_idx_words_list.append(p_a_idx[0]+":"+p_data[p_begin_idx:p_end_idx]+"("+p_a_idx[1]+")")
    return p_idx_words_list


file_path_name_list =[]
with open("practice_time_documents.txt", "r") as f:
    file_path_name_list = f.read().split('\n')

for file_path_name in file_path_name_list:
    if len(file_path_name) < 3:
        break
    file_path, file_name = os.path.split(file_path_name)
    file_real_path = os.path.join(real_dir, file_path, file_name, file_name + real_suffix)
    file_predict_path = os.path.join(predict_dir, file_path, file_name, file_name + predict_suffix)
    file_content_path = os.path.join(content_dir, file_path, file_name, file_name)
    with open(file_content_path, "r") as f:
        data = f.read()  # 读取文件
    s_1 = set(get_span_type_text(file_real_path))
    s_2 = set(get_span_type_text(file_predict_path))

    s_3 = s_1 & s_2
    s_real_last = s_1 - s_3
    s_predict_last = s_2 - s_3

    real_idx_words_list = get_words_by_idx(s_real_last, data)
    if len(real_idx_words_list) > 0:
        print("real:" + file_name)
        for one_real_idx_words in real_idx_words_list:
            print(one_real_idx_words)

    predict_idx_words_list = get_words_by_idx(s_predict_last, data)
    if len(predict_idx_words_list) > 0:
        print("predict:" + file_name)
        for one_predict_idx_words in predict_idx_words_list:
            print(one_predict_idx_words)



