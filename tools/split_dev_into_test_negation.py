import csv
import os

import pandas as pd
import tsv as tsv

split_file_input_path = 'practice_text/negation/dev.tsv'
split_file_labels_path = 'practice_text/negation/dev_labels.txt'

train_tsv_path = 'path/negation/train.tsv'
dev_tsv_path = 'path/negation/dev.tsv'
dev_label_tsv_path = 'path/negation/ref/negation/gold.tsv'
base_dir='path/negation/ref/negation'
os.makedirs(base_dir, exist_ok=True)

text_col_name = ["text"]
label_col_name = ["labels"]
input_text = pd.read_csv(split_file_input_path, header=None, names=text_col_name,
                         delimiter="\t", quoting=3)
input_labels = pd.read_csv(split_file_labels_path, header=None,
                           names=label_col_name, delimiter="\t", quoting=3)
text_list = []
label_list = []
for i, label in enumerate(input_labels["labels"]):
    label_list.append(label)
for i, text in enumerate(input_text["text"]):
    text_list.append(text)
num = len(label_list)
train_list = []
dev_list = []
dev_label_list = []
for i in range(num):
    if i % 5 == 6:
        dev_list.append([text_list[i]])
        dev_label_list.append([label_list[i]])
    else:
        train_list.append([label_list[i], text_list[i]])


with open(train_tsv_path, "w+") as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    # 写入多行用writerows
    writer.writerows(train_list)

with open(dev_tsv_path, "w+") as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    # 写入多行用writerows
    writer.writerows(dev_list)

with open(dev_label_tsv_path, "w+") as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    # 写入多行用writerows
    writer.writerows(dev_label_list)