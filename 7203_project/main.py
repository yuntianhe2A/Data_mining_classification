from _csv import writer
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import KFold
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)

df = pd.read_csv('Ecoli.csv')
# dataFrame for class 1
df1 = df.loc[df['Target (Col 107)'] == 1.0]
# dataFrame for class 0
df0 = df.loc[df['Target (Col 107)'] == 0.0]

# Pick several useful columns to perform classification instead of using all features.
var_columns = ['Nom (Col 106)', 'Num (Col 4)', 'Num (Col 95)', 'Num (Col 27)', 'Num (Col 35)', 'Num (Col 46)',
               'Num (Col 66)', 'Num (Col 102)']
target_column = 'Target (Col 107)'


def main():
    df_lof = pre_processing(df)  # remove outliers
    print("random forest cross validation result: ")
    # random_forest(df_lof)
    # predict_test(df_lof)
    testabc(df_lof)
    # decision_tree_cross_validation(df1)


def pre_processing(df):
    df_impute = impute(df)
    df_LOF = LOF_outliers(df_impute)
    # df_x = df_LOF.loc[:, var_columns]
    return df_LOF


def predict_test(train_df, input_test_file='Ecoli_test.csv', ):
    test_accuracy_avg = random_forest(train_df, score_metrics='accuracy')
    test_f1_avg = random_forest(train_df)
    cross_validation_results = [round(test_accuracy_avg, 3), round(test_f1_avg, 3)]
    print(cross_validation_results)
    X = train_df.loc[:, var_columns]
    y = train_df.loc[:, target_column]
    classifier = RandomForestClassifier(n_estimators=300, max_leaf_nodes=2, class_weight='balanced')

    classifier.fit(X, y)
    # read test
    df_test = pd.read_csv(input_test_file)
    print(len(df_test))
    df_test_X = df_test.loc[:, var_columns]
    predict_y = classifier.predict(df_test_X)
    predict_y_df = pd.DataFrame(predict_y)
    print(len(df_test))
    write_output(cross_validation_results, predict_y_df)
def testabc(train_df):

    X = train_df.loc[:, var_columns]
    y = train_df.loc[:, target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 20)
    classifier = RandomForestClassifier(n_estimators=300, max_leaf_nodes=2, class_weight='balanced')
    classifier.fit(X_train, y_train)
    predict_y = classifier.predict(X_test)
    print(cv_f1_scorer(classifier, X_test, y_test))
    X_test['predict']=predict_y
    X_test['ground_truth'] = y_test
    print(X_test)


def write_output(cross_validation_results, predict_y_df, output='s4722166.csv'):
    predict_y_df.to_csv(output, index=False,header=False)
    with open(output, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(cross_validation_results)
        f_object.close()


# f1 score : 0.8762
def random_forest(df, score_metrics='f1'):
    X = df.loc[:, var_columns]
    y = df.loc[:, target_column]
    classifier = RandomForestClassifier(n_estimators=300, max_leaf_nodes=2, class_weight='balanced')
    # clf.fit(X,y)
    if score_metrics == 'f1':
        print("F1 score:")
        kfold_scores = cross_validate(classifier, X, y, cv=10, scoring=cv_f1_scorer, return_train_score=True)
    else:
        print("Accuracy score:")
        kfold_scores = cross_validate(classifier, X, y, cv=10, scoring=cv_accuracy_scorer, return_train_score=True)
    #
    train_f1_avg = np.mean(kfold_scores['train_score'])
    test_f1_avg = np.mean(kfold_scores['test_score'])
    print("Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(train_f1_avg, test_f1_avg, train_f1_avg - test_f1_avg))
    return test_f1_avg


# # f1 score : 0.8762
# def decision_tree_cross_validation(df):
#     X = df.loc[:, var_columns]
#     y = df.loc[:, target_column]
#     for num_leaf_node in range(2, 16):
#         model_tree = DecisionTreeClassifier(max_leaf_nodes=num_leaf_node, class_weight='balanced')
#         kfold_scores = cross_validate(model_tree, X, y, cv=10, scoring=cv_f1_scorer, return_train_score=True)
#         # Find average train and test score
#         train_f1_avg = np.mean(kfold_scores['train_score'])
#         test_f1_avg = np.mean(kfold_scores['test_score'])
#         print("Nodes:{}, Train_f1:{:.4f}, Valid_f1:{:.4f}, Diff:{:.4f}".format(num_leaf_node, train_f1_avg, test_f1_avg,
#                                                                                train_f1_avg - test_f1_avg))
#     # by cross validation, already get max_leaf_nodes: 2


def LOF_outliers(df):
    df1 = df.loc[df['Target (Col 107)'] == 1.0]
    df0 = df.loc[df['Target (Col 107)'] == 0.0]

    # get outliers from all the instances of class 1
    X1 = df1.loc[:, var_columns]
    clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
    y1_pred = clf.fit_predict(X1)
    index = df1.index[:]
    dfs = pd.DataFrame(y1_pred, columns=['abc'], index=index)
    outliers1 = dfs.index[dfs['abc'] == -1]
    outlier_1_list = outliers1.tolist()

    # get outliers from all the instances of class 0
    X0 = df0.loc[:, var_columns]
    y0_pred = clf.fit_predict(X0)
    index_0 = df0.index[:]
    dfs = pd.DataFrame(y0_pred, columns=['abc'], index=index_0)
    outliers0 = dfs.index[dfs['abc'] == -1]
    outlier_0_list = outliers0.tolist()
    outlier_list = outlier_0_list + outlier_1_list
    print("Outliers to be removed from original dataset: ", outlier_list)
    df_no_outliers = df.drop(index=outlier_list)
    print("Total examples after removing outliers: ", len(df_no_outliers))
    return df_no_outliers


def cv_f1_scorer(model, X, y):
    return metrics.f1_score(y, model.predict(X))


def cv_accuracy_scorer(model, X, y):
    return metrics.accuracy_score(y, model.predict(X))


def impute(df):
    mean0 = df0.mean()
    mean1 = df1.mean()
    for i in range(1, 104):
        column_name = 'Num (Col ' + str(i) + ')'
        df.loc[(df[column_name].isna()) & (df['Target (Col 107)'] == 1.0), column_name] = mean1[i - 1]
        df.loc[(df[column_name].isna()) & (df['Target (Col 107)'] == 0.0), column_name] = mean0[i - 1]
    for j in range(104, 107):
        column_name = 'Nom (Col ' + str(j) + ')'
        df.loc[(df[column_name].isna()) & (df['Target (Col 107)'] == 0.0), column_name] = \
            df0[column_name].value_counts().index[0]
        df.loc[(df[column_name].isna()) & (df['Target (Col 107)'] == 1.0), column_name] = \
            df1[column_name].value_counts().index[0]
    return df


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
