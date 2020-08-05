#!/usr/bin/python
import os
import sys

# import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from df3.utils import config
from df3.utils.data_reader import FeatureDictionary, DataParser
from df3.utils.metrics import *

sys.path.append("..")
from df3.models.DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)
from sklearn.metrics import roc_auc_score

def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)

    cols = [c for c in dfTrain.columns if c not in ["label"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols]
    y_train = dfTrain["label"].values

    print(dfTrain.columns)
    return X_train, y_train


def _fit(dfTrain, folds, dfm_params,y_train):
    fd = FeatureDictionary(dfTrain=dfTrain,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train = data_parser.parse(df=dfTrain)


    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    _get = lambda x, l: [x[i] for i in l]


    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        y_pred = dfm.predict(Xi_valid_, Xv_valid_)
        print("auc-->" + str(roc_auc_score(y_valid_, y_pred)))




def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png" % model_name)
    plt.close()


def _predict(dfTrain, dfm_params, trainIndex):

    fd = FeatureDictionary(dfTrain=dfTrain,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train = data_parser.parse(df=dfTrain)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    dfm = DeepFM(**dfm_params)

    checkpoint_dir = os.path.abspath(os.path.join(dfm.model_dir, "checkpoints" + str(trainIndex)))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    dfm.restore(checkpoint_prefix)
    y_pred = dfm.predict(Xi_train, Xv_train)
    print("auc-->"+str(roc_auc_score(y_train,y_pred)))



if __name__ == "__main__":

    # load data
    X_train, y_train = _load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train.values, y_train))

    # ------------------ DeepFM Model ------------------
    # params
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 20,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 10,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "random_seed": config.RANDOM_SEED,
        "num_checkpoints": config.NUM_CK_POINTS,
        "model_dir": config.SUB_DIR,
        "checkpoint_every": config.CHECKPOINT_EVERY,
        "trainIndex" : 0
    }

    _fit(X_train,folds,dfm_params,y_train)#初始化训练
    #
    dfm_params ["trainIndex"] = 1  #第一次增量训练
    _fit(X_train, folds, dfm_params, y_train)
    #
    dfm_params["trainIndex"] = 2  #第二次增量训练
    _fit(X_train, folds, dfm_params, y_train)

    dfm_params["trainIndex"] = 2 #预测数据
    _predict(X_train , dfm_params, 2)
