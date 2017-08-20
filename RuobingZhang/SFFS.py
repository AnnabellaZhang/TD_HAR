# coding: utf-8
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def SFFS(X_train, y_train, X_val, y_val):
    column = X_train.columns
    ftr = np.array([])
    ind = np.zeros(column.shape, dtype=bool)
    max_score = 0

    count = 0

    while ftr.size < column.shape[1]:
        ## Forward
        max_score_forward = 0
        select_ftr = -1
        for i, col in enumerate(ind):
            if col:
                continue
            else:
                max_score_backward, select_ftr = evaFtr(X_train, y_train, X_val, y_val,
                                                        ftr, i, max_score_forward, select_ftr)
        if select_ftr >= 0:
            ftr = np.append(ftr, column[select_ftr])
            ind[select_ftr] = True

        ## Backward
        max_score_backward = max_score_forward
        while ftr.size > 0:
            select_ftr = -1
            for i, n in enumerate(ftr):
                max_score_backward, select_ftr = evaFtr(X_train, y_train, X_val, y_val,
                                                        ftr, i, max_score_backward, select_ftr)
            if select_ftr >= 0:
                ftr = np.delete(ftr, select_ftr)
                #ind[select_ftr] = False
            else:
                break

        count += 1
        print(count)

        if max_score < max_score_backward:
            max_score = max_score_backward
        else:
            break
    return ftr


def evaFtr(X_train, y_train, X_val, y_val, ftr, i, max_score, select_ftr):
        tmp_ftr = np.delete(ftr, i)
        tmp_X_train = X_train.loc[:, tmp_ftr]
        tmp_X_val = X_val.loc[:, tmp_ftr]
        cl = DecisionTreeClassifier()
        cl.fit(tmp_X_train, y_train)
        score = accuracy_score(y_val, cl.predict(tmp_X_val))
        if score > max_score:
            max_score = score
            select_ftr = i
        return max_score, select_ftr