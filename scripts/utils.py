import numpy as np

class ClassificationTreeNode:
    def __init__(self, X: np.array, feature_idx: int, feature_val: float, label_probs: list, information_gain: float, parent = None):
        """
        X: np.array, shape = (n_samples, n_features)
        feature_idx: int, index of the feature used to split the node
        feature_val: float, value of the feature used to split the node
        label_probs: list, list of probabilities of each class in the node
        information_gain: float, information gain of the split
        parent: ClassificationTreeNode, parent node
        """
        self.X = X
        self.feature_idx: int = feature_idx
        self.feature_val: float = feature_val
        self.label_probs: list = label_probs
        self.information_gain: float = information_gain
        self.parent: ClassificationTreeNode = parent
        self.left: ClassificationTreeNode = None
        self.right: ClassificationTreeNode = None

    def update_left(self, left):
        self.left = left

    def update_right(self, right):
        self.right = right

class RegressionTreeNode:
    def __init__(self, X: np.array, feature_idx: int, feature_val: float, parent = None):
        """
        X: np.array, shape = (n_samples, n_features)
        feature_idx: int, index of the feature used to split the node
        feature_val: float, value of the feature used to split the node
        parent: RegressionTreeNode, parent node
        """
        self.X = X
        self.feature_idx: int = feature_idx
        self.feature_val: float = feature_val
        self.parent: RegressionTreeNode = parent
        self.left: RegressionTreeNode = None
        self.right: RegressionTreeNode = None
        self.predict_val: float = None
    
    def update_left(self, left):
        self.left = left

    def update_right(self, right):
        self.right = right
