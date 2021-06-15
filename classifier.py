import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from config import Config
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

def combine(path, cols):
    total_df = pd.DataFrame
    minority = pd.read_csv('Data/minority_class', delimiter=',')
    minority.drop(['Unnamed: 0'], axis=1, inplace=True)
    if path == "centroids":
        df = pd.read_csv("Data/centroids.csv", delimiter=',')
        df.columns = cols
        total_df = pd.concat([df, minority])
    elif path == "random":
        df = pd.read_csv('Data/random.csv', delimiter=',')
        df.drop(['Unnamed: 0', 'cluster'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "one_neig":
        df = pd.read_csv('Data/one_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    elif path == "n_neig":
        df = pd.read_csv('Data/n_neig.csv', delimiter=',')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        total_df = pd.concat([df, minority])
    else:
        df = pd.read_csv("Data/numerical_data.csv", delimiter=',')
        total_df = df
    return total_df


def logistic_reg(df):
    class_name = ['negative', 'positive']
    X = df[Config.training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    ls_accuracy = []
    ls_precision = []
    ls_recall = []
    ls_f1 = []
    ls_auc = []
    for num_iter in np.linspace(10, 100, 90):
        logistic_regression = LogisticRegression(max_iter=num_iter)
        logistic_regression.fit(X_train, y_train)
        y_pred = logistic_regression.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        ls_accuracy.append(accuracy)
        precision = metrics.precision_score(y_test, y_pred)
        ls_precision.append(precision)
        recall = metrics.recall_score(y_test, y_pred)
        ls_recall.append(recall)
        f1 = metrics.f1_score(y_test, y_pred)
        ls_f1.append(f1)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
        auc = metrics.auc(fpr, tpr)
        ls_auc.append(auc)
    # logistic_regression = LogisticRegression(max_iter=100)
    # logistic_regression.fit(X_train, y_train)
    # y_pred = logistic_regression.predict(X_test)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    # auc = metrics.auc(fpr, tpr)
    # print("after 100 iterations: ", accuracy, precision, recall, f1, auc)
    # print(classification_report(y_test, y_pred, target_names=class_name))
    return ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc


def svm(df):
    X = df[Config.training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    ls_accuracy = []
    ls_precision = []
    ls_recall = []
    ls_f1 = []
    ls_auc = []
    for num_iter in np.linspace(10, 100, 90):
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=num_iter))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        ls_accuracy.append(accuracy)
        precision = metrics.precision_score(y_test, y_pred)
        ls_precision.append(precision)
        recall = metrics.recall_score(y_test, y_pred)
        ls_recall.append(recall)
        f1 = metrics.f1_score(y_test, y_pred)
        ls_f1.append(f1)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
        auc = metrics.auc(fpr, tpr)
        ls_auc.append(auc)
    return ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc


def nb(df):
    X = df[Config.training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    ls_accuracy = []
    ls_precision = []
    ls_recall = []
    ls_f1 = []
    ls_auc = []
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    ls_accuracy.append(accuracy)
    precision = metrics.precision_score(y_test, y_pred)
    ls_precision.append(precision)
    recall = metrics.recall_score(y_test, y_pred)
    ls_recall.append(recall)
    f1 = metrics.f1_score(y_test, y_pred)
    ls_f1.append(f1)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    ls_auc.append(auc)
    return ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc


def grad_boost(df):
    X = df[Config.training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    ls_accuracy = []
    ls_precision = []
    ls_recall = []
    ls_f1 = []
    ls_auc = []
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=20, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    ls_accuracy.append(accuracy)
    precision = metrics.precision_score(y_test, y_pred)
    ls_precision.append(precision)
    recall = metrics.recall_score(y_test, y_pred)
    ls_recall.append(recall)
    f1 = metrics.f1_score(y_test, y_pred)
    ls_f1.append(f1)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    ls_auc.append(auc)
    return ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc


if __name__ == '__main__':
    ls_paths = ["imbalanced", "centroids", "random",
                "one_neig", "n_neig"]
    # for name in ls_paths:
    #     total_df = combine(name, Config.cols)
    #     print(total_df.shape)
    #     ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = nb(total_df)
    #     print(ls_accuracy)
    #     plt.plot(np.linspace(10, 100, 90), ls_accuracy, label=f"{name}")
    #     plt.xlabel("iterations")
    #     plt.ylabel("accuracy")
    #     plt.legend()
    # plt.show()

    all_ls_accuracy = []
    all_ls_precision = []
    all_ls_recall = []
    all_ls_f1 = []
    all_ls_auc = []
    for name in ls_paths:
        total_df = combine(name, Config.cols)
        print(total_df.shape)
        ls_accuracy, ls_precision, ls_recall, ls_f1, ls_auc = nb(total_df)
        print(ls_accuracy)
        all_ls_accuracy.append(ls_accuracy[0])
        all_ls_precision.append(ls_precision[0])
        all_ls_recall.append(ls_recall[0])
        all_ls_f1.append(ls_f1[0])
        all_ls_auc.append(ls_auc[0])
    y_pos = np.arange(len(ls_paths))
    plt.bar(y_pos, all_ls_accuracy)
    plt.xticks(y_pos, ls_paths)
    plt.ylabel("accuracy")
    plt.show()
