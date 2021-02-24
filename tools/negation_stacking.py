import os
import pandas as pd
import numpy as np
import csv
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import precision_score, f1_score, recall_score
from torch import nn, tensor

x_train_1_dir = "negation_stacking/train_1/x_train_pred"
y_train_1_file = "negation_stacking/train_1/gold.tsv"

x_train_2_dir = "negation_stacking/train_2/x_train_pred"
y_train_2_file = "negation_stacking/train_2/gold.tsv"

x_train_3_dir = "negation_stacking/train_3/x_train_pred"
y_train_3_file = "negation_stacking/train_3/gold.tsv"

x_train_4_dir = "negation_stacking/train_4/x_train_pred"
y_train_4_file = "negation_stacking/train_4/gold.tsv"

x_train_5_dir = "negation_stacking/train_5/x_train_pred"
y_train_5_file = "negation_stacking/train_5/gold.tsv"

m = nn.Softmax(dim=1)


def get_train_data(x_train_dir, y_train_file):
    tsv_files = os.walk(x_train_dir)
    x = None
    # stacking pred
    for path, dir_list, file_list in tsv_files:
        for file_name in file_list:
            df = pd.read_csv(os.path.join(path, file_name), sep='\t', header=None)
            if x is None:
                x = m(tensor(df.values))
            else:
                x = np.hstack((x, m(tensor(df.values))))
    y = pd.read_csv(y_train_file, sep='\t', header=None)
    return x, y.values.flatten()


def hard_voting(x_train_dir, y_train_file):
    x, y = get_train_data(x_train_dir, y_train_file)
    b = x.argmax(axis=1)
    b_mod = (b % 2 - 0.5) * 2
    y_pre = None
    if y_pre is None:
        y_pre = b_mod
    else:
        y_pre = np.vstack(y_pre, b_mod)
    return y_pre, y


def soft_voting(x_train_dir, y_train_file):
    tsv_files = os.walk(x_train_dir)
    x = None
    # stacking pred
    for path, dir_list, file_list in tsv_files:
        for file_name in file_list:
            df = pd.read_csv(os.path.join(path, file_name), sep='\t', header=None)
            if x is None:
                x = df.values.argmax(axis=1)
            else:
                x = np.vstack((x, df.values.argmax(axis=1)))
    y = pd.read_csv(y_train_file, sep='\t', header=None)
    y_pre = x.T.sum(axis=1) > 3
    y_pre = y_pre.astype(int)
    y_pre = (y_pre - 0.5) * 2
    return y_pre, y.values.flatten()


def write_pred_tsv(y_test_pred, output_file):
    with open(output_file, "w+") as csv_file:
        writer = csv.writer(csv_file)
        # 写入多行用writerows
        for one_y_test_pred in y_test_pred.astype(int).tolist():
            writer.writerow([one_y_test_pred])


x_train_1, y_train_1 = get_train_data(x_train_1_dir, y_train_1_file)
x_train_2, y_train_2 = get_train_data(x_train_2_dir, y_train_2_file)
x_train_3, y_train_3 = get_train_data(x_train_3_dir, y_train_3_file)
x_train_4, y_train_4 = get_train_data(x_train_4_dir, y_train_4_file)
x_train, y_train = np.vstack((x_train_1, x_train_2)), np.append(y_train_1, y_train_2)
x_train, y_train = np.vstack((x_train, x_train_3)), np.append(y_train, y_train_3)
x_train, y_train = np.vstack((x_train, x_train_4)), np.append(y_train, y_train_4)
x_test, y_test = get_train_data(x_train_5_dir, y_train_5_file)

# clf = SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
#           decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
#           max_iter=-1, probability=False, random_state=None, shrinking=True,
#           tol=0.001, verbose=False)


# clf = RandomForestClassifier(n_estimators=14, criterion='gini', max_depth=None, min_samples_split=2,
#                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
#                              max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
#                              bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
#                              warm_start=False, class_weight=None)
#
# clf=tree.DecisionTreeClassifier()

# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
y_pred, y_test = hard_voting(x_train_1_dir, y_train_1_file)
# write_pred_tsv(y_pred,"system.tsv")
print('真实值：', y_test)
print('预测值：', y_pred)
print('f1：', f1_score(y_pred, y_test))
print('precision:', precision_score(y_pred, y_test))
print('recall:', recall_score(y_pred, y_test))
