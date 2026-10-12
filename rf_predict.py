"""纯 numpy 实现的 randomForest 分类预测器，读取 export_model.R 导出的 model.json。

与 R 中 predict(caret_train_obj, type="prob") 行为一致：
先做 caret 的 center/scale 标准化，再对每个样本遍历所有树，
按终端节点类别计数，得到每个类别的概率。
"""
import json
import os

import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")


class RandomForestModel:
    def __init__(self, path=_MODEL_PATH):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.levels = data["levels"]
        self.features = data["features"]
        self.mean = np.asarray(data["mean"], dtype=float)
        self.std = np.asarray(data["std"], dtype=float)
        self.trees = [
            {k: np.asarray(v) for k, v in tree.items()} for tree in data["trees"]
        ]

    def predict_proba(self, X):
        """X: (n_samples, n_features) 的数组，特征顺序须与 self.features 一致。
        返回 (n_samples, n_classes) 的概率矩阵，列顺序与 R 的 type='prob' 相同。"""
        X = np.asarray(X, dtype=float)
        # caret preProcess: center + scale
        Xs = (X - self.mean) / self.std
        n_samples = Xs.shape[0]
        n_classes = len(self.levels)
        votes = np.zeros((n_samples, n_classes), dtype=float)

        for tree in self.trees:
            nodestatus = tree["nodestatus"]
            bestvar = tree["bestvar"]
            xbestsplit = tree["xbestsplit"]
            nodepred = tree["nodepred"]
            left = tree["left"]
            right = tree["right"]
            nd = tree["ndbigtree"]

            # randomForest 4.7 约定：nodestatus 1=内部节点，-1=终端节点
            node = np.ones(n_samples, dtype=int)
            active = node <= nd
            while active.any():
                idx = np.where(active)[0]
                ns = nodestatus[node[idx] - 1]
                terminal = ns < 0
                if terminal.any():
                    ti = idx[terminal]
                    votes[ti, nodepred[node[ti] - 1] - 1] += 1
                    active[ti] = False
                if (~terminal).any():
                    ii = idx[~terminal]
                    v = bestvar[node[ii] - 1] - 1  # 转为 0-based 特征列
                    go_left = Xs[ii, v] <= xbestsplit[node[ii] - 1]
                    node[ii] = np.where(go_left, left[node[ii] - 1], right[node[ii] - 1])

        return votes / len(self.trees)


_model = None


def get_model():
    global _model
    if _model is None:
        _model = RandomForestModel()
    return _model
