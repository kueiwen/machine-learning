
class ClassificationTreeNode:
    def __init__(self, X, feature_idx, feature_val, label_probs, information_gain, parent = None):
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
    def __init__(self, X, feature_idx, feature_val, parent = None):
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
