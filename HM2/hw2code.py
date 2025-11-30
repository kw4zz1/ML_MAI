import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)
    n = len(y)
    if n <= 1:
        return np.array([]), np.array([]), None, None

    sort_idx = np.argsort(x)
    xs = x[sort_idx]
    ys = y[sort_idx]

    distinct = xs[1:] != xs[:-1]
    if not np.any(distinct):
        return np.array([]), np.array([]), None, None

    thresholds = (xs[:-1] + xs[1:]) / 2.0
    thresholds = thresholds[distinct]

    ys1 = (ys == 1).astype(int)
    cumsum1 = np.cumsum(ys1)

    indices = np.where(distinct)[0]
    left_counts = indices + 1
    left_ones = cumsum1[indices]
    left_p1 = left_ones / left_counts
    left_p0 = 1.0 - left_p1
    H_left = 1.0 - left_p1 ** 2 - left_p0 ** 2

    right_counts = n - left_counts
    right_ones = cumsum1[-1] - left_ones
    right_p1 = right_ones / right_counts
    right_p0 = 1.0 - right_p1
    H_right = 1.0 - right_p1 ** 2 - right_p0 ** 2

    ginis = - (left_counts / n) * H_left - (right_counts / n) * H_right

    best_idx = int(np.argmax(ginis))
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, float(threshold_best), float(gini_best)


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) == 0:
            node["type"] = "terminal"
            node["class"] = None
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0])
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        feature_best = None
        threshold_best = None
        gini_best = None
        split_mask = None

        n_features = sub_X.shape[1]
        for feature in range(n_features):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = np.asarray(sub_X[:, feature], dtype=float)
                categories_map = None
            elif feature_type == "categorical":
                cats, counts = np.unique(sub_X[:, feature], return_counts=True)
                ones = np.array([np.sum(sub_y[sub_X[:, feature] == c] == 1) for c in cats])
                prop = ones / counts.astype(float)
                order = np.argsort(prop)
                sorted_cats = cats[order]
                categories_map = {cat: i for i, cat in enumerate(sorted_cats)}
                feature_vector = np.array([categories_map[val] for val in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            thresholds, ginis, thr_best, gini_thr_best = find_best_split(feature_vector, sub_y)
            if thresholds.size == 0:
                continue

            valid_mask = np.ones_like(ginis, dtype=bool)
            if self._min_samples_leaf is not None:
                left_counts = np.sum(feature_vector[:, None] < thresholds[None, :], axis=0)
                right_counts = len(feature_vector) - left_counts
                valid_mask &= (left_counts >= self._min_samples_leaf) & (right_counts >= self._min_samples_leaf)

            if not np.any(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_ginis = ginis[valid_indices]
            max_g = np.max(valid_ginis)
            cand_idx = valid_indices[np.where(valid_ginis == max_g)[0]]
            chosen = cand_idx[np.argmin(thresholds[cand_idx])]
            chosen_threshold = float(thresholds[chosen])
            chosen_gini = float(ginis[chosen])

            if gini_best is None or chosen_gini > gini_best:
                feature_best = feature
                gini_best = chosen_gini
                threshold_best = chosen_threshold
                if feature_type == "real":
                    split_mask = feature_vector < threshold_best
                else:
                    categories_split = [cat for cat, idx in categories_map.items() if idx < threshold_best]
                    split_mask = np.array([val in categories_split for val in sub_X[:, feature]])
                    threshold_best = categories_split

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        node["type"] = "nonterminal"
        node["feature_split"] = int(feature_best)
        if self._feature_types[feature_best] == "real":
            node["threshold"] = float(threshold_best)
        else:
            node["categories_split"] = list(threshold_best)

        node["left_child"] = {}
        node["right_child"] = {}

        left_X = sub_X[split_mask]
        left_y = sub_y[split_mask]
        right_X = sub_X[np.logical_not(split_mask)]
        right_y = sub_y[np.logical_not(split_mask)]

        self._fit_node(left_X, left_y, node["left_child"], depth + 1)
        self._fit_node(right_X, right_y, node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node.get("type") == "terminal":
            return node.get("class")

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            thr = node["threshold"]
            if x[feature] < thr:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            cats = node.get("categories_split", [])
            if x[feature] in cats:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_depth(self, node=None):
        if node is None:
            node = self._tree
        
        if node.get("type") == "terminal":
            return 0
        
        left_depth = self.get_depth(node.get("left_child", {}))
        right_depth = self.get_depth(node.get("right_child", {}))
        
        return 1 + max(left_depth, right_depth)
    
    def get_n_leaves(self, node=None):
        if node is None:
            node = self._tree
        
        if node.get("type") == "terminal":
            return 1
        
        left_leaves = self.get_n_leaves(node.get("left_child", {}))
        right_leaves = self.get_n_leaves(node.get("right_child", {}))
        
        return left_leaves + right_leaves
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (required for sklearn compatibility)."""
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (required for sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, f'_{key}', value)
        return self