import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def logistic_reg(path):
    df = pd.read_csv(path, delimiter=',')
    training_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
                     'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                     'job_management', 'job_retired', 'job_self-employed', 'job_services',
                     'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                     'marital_divorced', 'marital_married', 'marital_single',
                     'marital_unknown', 'contact_cellular', 'contact_telephone',
                     'default_no', 'default_unknown', 'default_yes', 'housing_no',
                     'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
                     'education_cat', 'poutcome_cat']
    X = df[training_cols]
    y = df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    return accuracy, precision, recall, f1, auc


if __name__ == '__main__':

    # ori_path = "Data/numerical_data.csv"
    # ori_accuracy, ori_precision, ori_recall, ori_f1, ori_auc = logistic_reg(ori_path)
    # print(ori_accuracy, ori_precision, ori_recall, ori_f1, ori_auc)

    # 1
    minority_path = 'Data/minority_class'
    # cen_path = "Data/centers.csv"

    # df = pd.read_csv(cen_path, delimiter=',')
    cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
            'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
            'job_management', 'job_retired', 'job_self-employed', 'job_services',
            'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
            'marital_divorced', 'marital_married', 'marital_single',
            'marital_unknown', 'contact_cellular', 'contact_telephone',
            'default_no', 'default_unknown', 'default_yes', 'housing_no',
            'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
            'education_cat', 'poutcome_cat', 'y_cat']
    # print(df.shape)
    # df.columns = cols  # undersampled majority class
    #
    minority = pd.read_csv(minority_path, delimiter=',')
    minority.drop(['Unnamed: 0'], axis=1, inplace=True)
    minority.columns = cols
    print(minority.shape)
    # total_df = pd.concat([df, minority])
    # print(total_df)
    # print(cols[:-1])
    # X = total_df[cols[:-1]]
    # y = total_df['y_cat']
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #
    # logistic_regression = LogisticRegression()
    # logistic_regression.fit(X_train, y_train)
    # y_pred = logistic_regression.predict(X_test)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    # auc = metrics.auc(fpr, tpr)
    # print(accuracy, precision, recall, f1, auc)

    # 2
    # random_path = 'Data/random.csv'
    # df = pd.read_csv(random_path, delimiter=',')
    # df.drop(['Unnamed: 0', 'cluster'], axis=1, inplace=True)
    # print(df.shape)
    # total_df = pd.concat([df, minority])
    # print(total_df)
    # X = total_df[cols[:-1]]
    # y = total_df['y_cat']
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #
    # logistic_regression = LogisticRegression()
    # logistic_regression.fit(X_train, y_train)
    # y_pred = logistic_regression.predict(X_test)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    # auc = metrics.auc(fpr, tpr)
    # print(accuracy, precision, recall, f1, auc)

    # 3
    # one_neig_path = 'Data/one_neig.csv'
    # df = pd.read_csv(one_neig_path, delimiter=',')
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # print(df.shape)
    # total_df = pd.concat([df, minority])
    # print(total_df)
    # X = total_df[cols[:-1]]
    # y = total_df['y_cat']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #
    # logistic_regression = LogisticRegression()
    # logistic_regression.fit(X_train, y_train)
    # y_pred = logistic_regression.predict(X_test)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    # auc = metrics.auc(fpr, tpr)
    # print(accuracy, precision, recall, f1, auc)

    # 4
    n_neig_path = 'Data/n_neig.csv'
    df = pd.read_csv(n_neig_path, delimiter=',')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(df)
    total_df = pd.concat([df, minority])
    print(total_df)
    X = total_df[cols[:-1]]
    y = total_df['y_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    print(accuracy, precision, recall, f1, auc)

