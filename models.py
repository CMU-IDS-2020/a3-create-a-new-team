import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

categoricals = ["TYPE", "SEX", "GSM", "HAIR", "EYE", "ALIGN", "ID"]
numericals = ["APPEARANCES", "YEAR"]

X_col = ["TYPE", "GSM", "SEX", "HAIR", "APPEARANCES", "YEAR", "ALIGN", "ID"]
y_col = ["EYE"]


def set_seed(seed):
    np.random.seed(seed)


def encode_categorical(col):
    return LabelEncoder().fit_transform(col)


def load_data(directory="./archive"):
    path_dc = os.path.join(directory, 'dc-wikia-data.csv')
    path_marvel = os.path.join(directory, 'marvel-wikia-data.csv')
    # the Auburn Hair and Year issues have resolved in the csv file
    dc, marvel = pd.read_csv(path_dc), pd.read_csv(path_marvel)
    return merge(dc, marvel)


def merge(dc, marvel):
    dc['TYPE'] = 'DC'
    marvel['TYPE'] = 'Marvel'
    marvel.replace({"SEX": {"Genderfluid Characters": "Genderless Characters",
                            "Agender Characters": "Genderless Characters"}},
                   inplace=True)
    dc.replace({"SEX": {"Genderfluid Characters": "Genderless Characters",
                        "Agender Characters": "Genderless Characters"}},
               inplace=True)
    dc.loc[dc['SEX'] == "Transgender Characters",
           "GSM"] = "Transgender Characters"
    dc.loc[dc['SEX'] == "Transgender Characters", "SEX"] = np.nan
    data = pd.concat([dc, marvel])
    # fill NaNs in X-categorical columns
    data[list(set(categoricals) - set(y_col))] = data[list(set(categoricals) - set(y_col))].fillna('N/A')
    data = data[categoricals + numericals]
    # filter rows which has NaNs in (y cols | numerical cols)
    data = data.dropna()
    return data


def per_response_var(response_var, data, seed):
    set_seed(seed)
    all_col = ["TYPE", "GSM", "SEX", "EYE", "HAIR", "APPEARANCES", "YEAR", "ALIGN", "ID"]
    y_col = [response_var]
    all_col.remove(response_var)
    X_col = all_col
    data = data.copy()
    X = data[X_col]
    y = data[y_col]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, y_train = X, y
    # print(len(X_train), len(X_test))
    to_transform = list(set(numericals).intersection(set(y_col + X_col)))
    if len(to_transform) > 0:
        sc = StandardScaler()
        features = X_train[to_transform]
        scaler = sc.fit(features.values)
        features = scaler.transform(features.values)
        X_train.loc[:, to_transform] = features
        # X_test.loc[:, to_transform] = sc.transform(X_test[to_transform])
    # print(X_train[:5])
    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    d = {"Importance": clf.feature_importances_, "Variable": X_col}
    feature_importances = pd.DataFrame(data=d)
    # print(feature_importances)
    return feature_importances


def calc_feature_importances():
    seed = 0
    data = load_data()
    data = data.apply(lambda x: encode_categorical(x) if x.name in categoricals else x)
    res = {}

    if "APPEARANCES" in data.columns:
        data.loc[:, "APPEARANCES"] = np.log(data["APPEARANCES"] + 1)

    for target in ["SEX", "EYE", "HAIR"]:
        res[target] = per_response_var(target, data, seed)
    return res


if __name__ == "__main__":
    calc_feature_importances()
