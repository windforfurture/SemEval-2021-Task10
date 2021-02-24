import os
from bs4 import BeautifulSoup
import anafora
import numpy as np

model_name = ["c1",
              "c2",
              "c3",
              "c4",
              "c5"]

result_dir = "all-result"
file_base_dir = "submission/res/time"
all_bert_annotations_list = []

output_file_path = []

for model_name_one in model_name[1:]:
    result_bert_dir = os.path.join(result_dir, model_name_one, "time")
    text_directory_files = anafora.walk(result_bert_dir, xml_name_regex=".*((?<![.].{3})|[.]xml)$")
    bert_annotations_list = []
    one_output_file_path = []
    for text_files in text_directory_files:
        text_subdir_path, text_doc_name, text_file_names = text_files
        one_output_file_path.append([text_subdir_path, text_doc_name])
        anafora_file_path = os.path.join(result_bert_dir, text_subdir_path, text_file_names[0])
        one_data = anafora.AnaforaData.from_file(anafora_file_path)
        annotations = dict()
        for annotation in one_data.annotations:
            label = annotation.type
            for span in annotation.spans:
                start, end = span
                annotations[str(start) + ':' + str(end) + ':' + label] = (end, label)
        bert_annotations_list.append(annotations)
    if len(output_file_path) == 0:
        output_file_path.append(one_output_file_path)
    all_bert_annotations_list.append(bert_annotations_list)
output_file_path=output_file_path[0]
vote_result = []
for i in range(len(all_bert_annotations_list[0])):
    vote_result.append(dict())
for one_bert_annotations_list in all_bert_annotations_list:
    for i in range(len(one_bert_annotations_list)):
        annotation_keys = one_bert_annotations_list[i].keys()
        for one_key in annotation_keys:
            if vote_result[i].get(one_key) is None:
                vote_result[i][one_key] = 1
            else:
                key_num = vote_result[i].get(one_key)
                vote_result[i][one_key] = key_num + 1


def add_entity(data, doc_name, vote_one_result):
    anafora.AnaforaEntity()
    entity = anafora.AnaforaEntity()
    num_entities = len(data.xml.findall("annotations/entity"))
    entity.id = "%s@%s" % (num_entities, doc_name)
    vote_one_result_items = vote_one_result.split(":")
    entity.spans = ((int(vote_one_result_items[0]), int(vote_one_result_items[1])),)
    entity.type = vote_one_result_items[2]
    data.annotations.append(entity)


for i in range(len(output_file_path)):
    data = anafora.AnaforaData()
    doc_sub_path, doc_name = output_file_path[i]
    for key in vote_result[i]:
        if int(vote_result[i].get(key)) > 2:
            add_entity(data, doc_name, key)
    doc_path = os.path.join(file_base_dir, doc_sub_path)
    os.makedirs(doc_path, exist_ok=True)
    doc_path = os.path.join(doc_path, "%s.TimeNorm.system.completed.xml" % doc_name)
    data.to_file(doc_path)

