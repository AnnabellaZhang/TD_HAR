# coding: utf-8
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# Sequential Floating Forward Selection, the evaluation is based on Random Forest
def SFFS(X_train, y_train, X_val, y_val):
    column = X_train.columns
    ftr = np.array([])
    ind = np.zeros(column.shape, dtype=bool)
    max_score = 0

    count = 0

    while ftr.size < column.shape[0] and max_score < 1:
        ## Forward
        max_score_forward = max_score
        select_ftr = -1
        for i, col in enumerate(ind):
            if col:
                continue
            else:
                tmp_ftr = np.append(ftr, column[i])
                max_score_forward, select_ftr = evaFtr(X_train, y_train, X_val, y_val,
                                                       tmp_ftr, i, max_score_forward, select_ftr)
        if select_ftr >= 0:
            ftr = np.append(ftr, column[select_ftr])
            ind[select_ftr] = True

        ## Backward
        max_score_backward = max_score_forward
        while ftr.size > 1:
            select_ftr = -1
            for i, n in enumerate(ftr):
                tmp_ftr = np.delete(ftr, i)
                max_score_backward, select_ftr = evaFtr(X_train, y_train, X_val, y_val,
                                                        tmp_ftr, i, max_score_backward, select_ftr)
            if select_ftr >= 0:
                ftr = np.delete(ftr, select_ftr)
                # ind[select_ftr] = False
            else:
                break

        count += 1
        print('Loop = %d, Maximum Score = %f, Feature Num = %s' % (count, max(max_score, max_score_backward), ftr.size))

        if max_score < max_score_backward:
            max_score = max_score_backward
        else:
            print('Feature Selection Completed!')
            break

    return ftr


def evaFtr(X_train, y_train, X_val, y_val, tmp_ftr, i, max_score, select_ftr):
    tmp_X_train = X_train.loc[:, tmp_ftr]
    tmp_X_val = X_val.loc[:, tmp_ftr]
    cl = RandomForestClassifier()
    cl.fit(tmp_X_train, y_train)
    score = accuracy_score(y_val, cl.predict(tmp_X_val))
    if score > max_score:
        max_score = score
        select_ftr = i
    return max_score, select_ftr