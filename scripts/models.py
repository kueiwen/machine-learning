from typing import Tuple, List
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
from statistics import mode
from scripts.utils import ClassificationTreeNode, RegressionTreeNode

class LinearRegression:
    def __init__ (self, lr: float = 0.01, epochs: int = 1000):
        """
        Args:
            lr (float): learning rate
            epochs (int): number of epochs
        """
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: np.array, Y: np.array) -> Tuple[List, float]:
        """Train the Linear Regression model
        Args:
            X (np.array): feature matrix
            Y (np.array): target vector
        Returns:
            np.array: weights
            float: bias
        """
        self.X = X
        self.Y = Y
        if X.ndim == 1:
            self.n_features = 1
            self.X = X.reshape(-1, len(X))
        else:
            self.n_features = len(X.T)
            self.X = X
        self.w = np.zeros(self.n_features)
        self.b = 0
        prev_rmse = 0
        rmse = 1
        i = 0
        while abs(prev_rmse - rmse) > 0.000001:
            prev_rmse = rmse
            Y_pred = self.w.dot(self.X) + self.b
            error = self.Y - Y_pred
            D_w = -2/self.n_features * np.sum(X * error)
            D_b = -2/self.n_features * np.sum(error)
            self.w = self.w - self.lr * D_w
            self.b = self.b - self.lr * D_b
            rmse = np.sqrt(np.sum(error**2))
            i += 1
        return self.w, self.b
    
    def predict(self, X: np.array) -> np.array:
        """Predict the target vector
        Args:
            X (np.array): feature matrix
        Returns:
            np.array: predicted target vector
        """
        return self.w.dot(X) + self.b


class SVM:
    def __init__(self, C: float = 0.1, lr: float = 0.001, epochs: int = 10000):
        """
        Args:
            C (float): regularization parameter
            lr (float): learning rate
            epochs (int): number of epochs
        """
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.loss = None

    def fit(self, X, Y) -> Tuple[List, float]:
        """Train the SVM model
        Args:
            X (np.array): feature matrix
            Y (np.array): target vector
        Returns:
            np.array: weights
            float: bias
        """
        if X.ndim == 1:
            self.n_features = 1
            self.X = X.reshape(-1, len(X))
        else:
            self.n_features = len(X.T)
            self.X = X
        self.Y = Y
        self.w  = np.zeros(self.n_features)
        self.b = 0
        for i in range(self.epochs):
            self.loss = 0
            #np.random.shuffle(arr)
            for k in range(len(X)):
                if Y[k] * (self.w.dot(X[k]) + self.b) < 1:
                    self.loss += 0.5 * self.w.T.dot(self.w) + 1 - Y[k] * (self.w.dot(X[k]) + self.b)
                    self.w -= self.lr * (self.w - self.C * X[k].dot(Y[k]))
                    self.b -= self.lr * (-self.C * Y[k])
                else:
                    self.loss += 0.5 * self.w.T.dot(self.w)

        return self.w, self.b

    def getLoss(self) -> int:
        """Get the loss value
        Returns:
            int: loss value
        """
        return self.loss[0]
    
    def predict(self, X) -> np.array:
        """Predict the target vector
        Args:
            X (np.array): feature matrix
        Returns:
            np.array: predicted target vector
        """
        return self.w.dot(X)+self.b
    
    def plot(self, xlabel: str = '', ylabel: str = ''):
        """Plot the decision boundary
        Args:
            xlabel (str): x-axis label
            ylabel (str): y-axis label
        """
        colormap = np.array(['#e9c46a','#2a9d8f'])
        Y = np.array([0 if x==-1 else 1 for x in list(self.Y)])
        plt.scatter(self.X.T[0], self.X.T[1],c=colormap[Y])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.legend(loc="upper left")
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        l = []
        for i in range(1, int(math.sqrt(len(Y)))):
            if len(self.Y)%i == 0:
                l.append([i, len(Y)/i])
        
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], int(l[-1][0])), np.linspace(ylim[0], ylim[1], int(l[-1][1])))
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = (np.dot(xy, self.w) + self.b).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

class SVR:
    def __init__(self, lr: float = 0.001, epsilon: float = 0.01, epochs: int = 1000):
        """
        Args:
            lr (float): learning rate
            epsilon (float): epsilon value
            epochs (int): number of epochs
        """
        self.lr = lr
        self.epsilon = epsilon
        self.epochs = epochs

    def fit(self, X, Y) -> Tuple[List, float]:
        """Train the SVR model
        Args:
            X (np.array): feature matrix
            Y (np.array): target vector
        Returns:
            np.array: weights
            float: bias
        """
        if X.ndim == 1:
            self.n_features = 1
            self.X = X.reshape(-1, len(X))
        else:
            self.n_features = len(X.T)
            self.X = X
        self.Y = Y
        self.w  = np.zeros(self.n_features)
        self.b = 0
        
        i = 0
        for i in range(self.epochs):
            self.loss = 0
            y_pred = self.w.dot(self.X) + self.b
            for k in range(len(y_pred)):
                if Y[k] - y_pred[k] > 0 :
                    self.loss += 0.5 * self.w ** 2 + (Y[k] - y_pred[k]) - self.epsilon
                    self.w -= self.lr * (self.w - self.X.T[k])
                    self.b -= self.lr * (-1)
                else:
                    self.loss += 0.5 * self.w ** 2 - (Y[k] - y_pred[k]) - self.epsilon
                    self.w -= self.lr * (self.w + self.X.T[k])
                    self.b -= self.lr * (1)
        i += 1
        return self.w, self.b
            
    def getLoss(self) -> int:
        """Get the loss value
        Returns:
            int: loss value
        """
        return self.loss[0]
    
    def predict(self, X) -> np.array:
        """Predict the target vector
        Args:
            X (np.array): feature matrix
        Returns:
            np.array: predicted target vector
        """
        return self.w.dot(X)+self.b
    
    def plot(self, xlabel: str = '', ylabel: str = ''):
        """Plot the decision boundary
        Args:
            xlabel (str): x-axis label
            ylabel (str): y-axis label
        """
        self.y_pred = self.w * self.X + self.b
        plt.scatter(self.X, self.Y) 
        plt.plot([min(self.X[0]), max(self.X[0])], [min(self.y_pred[0]), max(self.y_pred[0])], color='red')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

class NB:
    def __init__(self):
        pass

    def gaussian_distribution(self, X: np.array, mean: np.array, var: np.array) -> np.array:
        """get likelihood based on gaussian distribution
        Args:
            X (np.array): data
            mean (np.array): mean of each feature
            var (np.array): variance of each feature
        Returns:
            likelihood of each feature for each data point"""
        return 1 / np.sqrt(var * 2 * math.pi) * np.exp(-0.5 * (X - mean) ** 2 / var)
    
    def fit(self, X: np.array, y: np.array) -> None:
        """Train Naive Bayes model
        Args:
            X (np.array): training data
            y (np.array): training labels
        """
        n_samples, n_features = X.shape
        self.n_classes = y.nunique()

        self.mean_cf = np.zeros((self.n_classes, n_features))
        self.variance_cf = np.zeros((self.n_classes, n_features))
        self.prior_c = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            X_c = X[y==c]

            self.mean_cf[c] = X_c.mean(axis=0)
            self.variance_cf[c] = X_c.var(axis=0)
            self.prior_c[c] = X_c.shape[0] / n_samples


    def predict(self, X: np.array) -> np.array:
        """Predict the testing data labels
        Args:
            X (np.array): testing data
        Returns:
            y_pred (np.array): predicted labels"""
        y_pred = []
        for x in X:
            posteriors = []

            for c in range(self.n_classes):
                likelihood = np.sum(self.gaussian_distribution(x, self.mean_cf[c], self.variance_cf[c]))
                prior = self.prior_c[c]
                posterior = likelihood * prior
                posteriors.append(posterior)

            y_pred.append(np.argmax(posteriors))
        
        return y_pred


class DecisionTreeClassifier:
    def __init__(self, criterion: str = 'entropy', max_depth: int = 6, min_samples_leaf: int = 1):
        """
        Args:
            criterion (str): criterion to split the data
            max_depth (int): maximum depth of the tree
            min_samples_leaf (int): minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree: DecisionTreeClassifier = None

    def entropy(self, class_probabilitues: list) -> float:
        """Implement the entropy function
        Args:
            class_probabilitues (list): A list of class probabilities
        Returns:
            entropy (float): The entropy of the given class probabilities"""
        return sum([-p * np.log2(p+1e-10) for p in class_probabilitues])

    def gini(self, class_probabilities: list) -> float:
        """Implement the gini function
        Args:
            class_probabilities (list): A list of class probabilities
        Returns:
            gini (float): The gini of the given class probabilities"""
        return 1 - sum([p**2 for p in class_probabilities])
    
    def split(self, X: np.array, y: np.array, feature_idx: int, feature_val: float) -> Tuple[np.array, np.array]:
        """Split the data into two group with the given feature and value
        Args:
            X (np.array): The input data
            y (np.array): The target labels
            feature_idx (int): The index of the feature to split
            feature_val (float): The value to split on
        Returns:
            x1 (np.array): left data
            x2 (np.array): right data
            y1 (np.array): left labels
            y2 (np.array): right labels
            p1 (np.array): The probability of each class in the left split data
            p2 (np.array): The probability of each class in the right split data"""
        x1 = X[X.T[feature_idx]<feature_val]
        x2 = X[X.T[feature_idx]>=feature_val]
        y1 = y[X.T[feature_idx]<feature_val]
        y2 = y[X.T[feature_idx]>=feature_val]
        n_classes = len(np.unique(y))

        if len(y1) == 0:
            p1 = [0]*n_classes
        else:
            p1 = [np.sum(y1==c)/len(y1) for c in range(n_classes)]
        if len(y2) == 0:
            p2 = [0]*n_classes
        else:
            p2 = [np.sum(y2==c)/len(y2) for c in range(n_classes)]

        return x1, x2, y1, y2, p1, p2

    def find_best_split(self, X: np.array, y: np.array) -> Tuple[int, float, float, np.array, np.array, np.array, np.array]:
        """Find the best split for the given data and criterion
        Args:
            X (np.array): The input data
            y (np.array): The target labels
        Returns:
            feature_idx (int): The index of the feature to split
            feature_val (float): The value to split on
            best_score (float): The best score of the split
            x1 (np.array): The left split data
            x2 (np.array): The right split data
            y1 (np.array): The left split target labels
            y2 (np.array): The right split target labels
            """
        n_features = X.shape[1]
        best_score = 10e+10
        flag = False
        for feature_idx in range(n_features):
            for feature_val in np.unique(X.T[feature_idx]):
                x1, x2, y1, y2, p1, p2 = self.split(X, y, feature_idx, feature_val)
                current_split = {'feature_idx': feature_idx, 'feature_val': feature_val, 
                                'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'best_score': best_score, 'p1': p1, 'p2': p2}
                if self.criterion == 'entropy':
                    score = self.entropy(p1) + self.entropy(p2)
                elif self.criterion == 'gini':
                    score = self.gini(p1) + self.gini(p2)
                if score < best_score:
                    flag = True
                    best_score = score
                    best_split = current_split
        if flag:
            return best_split['feature_idx'], best_split['feature_val'], best_split['best_score'], \
                    best_split['x1'], best_split['x2'], best_split['y1'], best_split['y2']
        else:
            return current_split['feature_idx'], current_split['feature_val'], current_split['best_score'], \
                    current_split['x1'], current_split['x2'], current_split['y1'], current_split['y2']
            
    def get_data_prob(self, y: np.array) -> np.array:
        """Get the probability of each class in the given data
        Args:
            y (np.array): The target labels
        Returns:
            label_probs (np.array): The probability of each class in the given data"""
        label_probs = np.zeros(self.total_nclasses, dtype = float)
        for label in range(self.total_nclasses):
            label_probs[label] = np.sum(y == label) / len(y)
        return label_probs

    def fit(self, X: np.array, y: np.array, current_depth: int = 0, parent = None) -> ClassificationTreeNode:
        """Build a decision tree for the given data
        Args:
            X (np.array): The input data
            max_depth (int): The maximum depth of the tree
        Returns:
            root (ClassificationTreeNode): The root node of the decision tree"""
        
        if current_depth == 0:
            self.total_nclasses = len(np.unique(y))

        if current_depth > self.max_depth:
            return None
        
        feature_idx, feature_val, best_score, x1, x2, y1, y2 = self.find_best_split(X, y)
        label_probs = self.get_data_prob(y)
        if self.criterion == 'entropy':
            node_info = self.entropy(label_probs)
        elif self.criterion == 'gini':
            node_info = self.gini(label_probs)
        information_gain = node_info - best_score
        tree = ClassificationTreeNode(X, feature_idx, feature_val, label_probs, information_gain, parent)

        if len(x1) < self.min_samples_leaf or len(x2) < self.min_samples_leaf:
            self.tree = tree
            return self.tree
        #print(current_depth)
        current_depth += 1
        tree.left = (self.fit(x1, y1, current_depth, parent = tree))
        tree.right = (self.fit(x2, y2, current_depth, parent = tree))

        self.tree = tree
        return self.tree
    
    def predict(self, X: np.array) -> np.array:
        """Predict the class of the given input data
        Args:
            X (np.array): The input data
        Returns:
            predictions (np.array): The predicted class of the input data"""
        predictions = []
        self.pred_prob = []
        for x in X:
            node = self.tree
            while node.left is not None and node.right is not None:
                if x[node.feature_idx] < node.feature_val:
                    node = node.left
                else:
                    node = node.right
            self.pred_prob.append(node.label_probs)
            predictions.append(np.argmax(node.label_probs))
        return np.array(predictions)
    
    def get_pred_prob(self, X: np.array) -> list:
        """Get the probability of each class for the given input data
        Args:
            X (np.array): The input data
        Returns:
            pred_prob (list): The probability of each class for the input data"""
        self.predict(X)
        return self.pred_prob
    
    def print_tree(self, Node: ClassificationTreeNode, depth: int = 0):
        """Print the decision tree
        Args:
            Node (ClassificationTreeNode): The root node of the tree
            depth (int): The depth of the tree"""
        if depth == 0:
            Node = self.tree
        if Node is None:
            return
        print(' | '*depth,'-', Node.feature_idx, '<' , Node.feature_val)
        if Node.left is None and Node.right is None:
            print('   '*depth, '| - class :', np.argmax(Node.label_probs))
        self.print_tree(Node.left, depth+1)
        print(' | '*depth,'-', Node.feature_idx, '>=' , Node.feature_val)
        if Node.left is None and Node.right is None:
            print('   '*depth, '| - class: ', np.argmax(Node.label_probs))
        self.print_tree(Node.right, depth+1)

    def calculate_feature_importance(self, node):
        """Calculate the feature importance of the tree
        Args:
            node (TreeNode): The root node of the tree
        Returns:
            feature_importances (dict): The feature importance of the tree
        """
        if node is None:
            return
        if node.left is None and node.right is None:
            return
        if node.left is not None:
            self.feature_importances[node.feature_idx] += node.information_gain
            self.calculate_feature_importance(node.left)
        if node.right is not None:
            self.feature_importances[node.feature_idx] += node.information_gain
            self.calculate_feature_importance(node.right)
        return self.feature_importances


class DecisionTreeRegressor:
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 1):
        """
        Args:
            max_depth (int): maximum depth of the tree
            min_samples_leaf (int): minimum number of samples required to be at a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree: DecisionTreeRegressor = None

    def mse(self, y_left: np.array, y_right: np.array) -> Tuple[float, float]:
        """Implement the mse function
        Args:
            y_left (np.array): An array of y values in left group
            y_right (np.array): An array of y values in right group
        Returns:
            mes_left (float): The mse of the given y_left
            mes_right (float): The mse of the given y_right"""
        mes_left = np.mean([(y_left[i]-np.mean(y_left))**2 for i in range(len(y_left))])
        mes_right = np.mean([(y_right[i]-np.mean(y_right))**2 for i in range(len(y_right))])
        return mes_left, mes_right
    
    def split(self, X: np.array, y: np.array, feature_idx: int, feature_val: float) -> Tuple[np.array, np.array]:
        """Split the data into two group with the given feature and value
        Args:
            X (np.array): The input data
            y (np.array): The target labels
            feature_idx (int): The index of the feature to split
            feature_val (float): The value to split on
        Returns:
                X_left (np.array): The probability of each class in the left split data
                X_right (np.array): The probability of each class in the right split data"""
        x1 = X[X.T[feature_idx]<feature_val]
        x2 = X[X.T[feature_idx]>=feature_val]
        y1 = y[X.T[feature_idx]<feature_val]
        y2 = y[X.T[feature_idx]>=feature_val]

        return x1, x2, y1, y2


    def get_cost(self, y1, y2):
        """Calculate the cost of the split
        Args:
            y1 (np.array): The target labels in the left split
            y2 (np.array): The target labels in the right split
        Returns:
            cost (float): The cost of the split"""
        mse_left, mse_right = self.mse(y1, y2)
        return (len(y1)/(len(y1)+len(y2))) * mse_left + (len(y2)/(len(y1)+len(y2))) * mse_right


    def find_best_split(self, X: np.array, y: np.array) -> Tuple[int, float, float, np.array, np.array, np.array, np.array]:
        """Find the best split for the given data and criterion
        Args:
            X (np.array): The input data
            y (np.array): The target labels
        Returns:
            feature_idx (int): The index of the feature to split
            feature_val (float): The value to split on
            best_score (float): The best score of the split
            x1 (np.array): The left split data
            x2 (np.array): The right split data
            y1 (np.array): The left split target labels
            y2 (np.array): The right split target labels
            """
        n_features = X.shape[1]
        best_score = 10e+10
        flag = False
        for feature_idx in range(n_features):
            all_val = np.unique(X.T[feature_idx])
            val = np.unique([np.percentile(all_val, 10),np.percentile(all_val, 20),np.percentile(all_val, 30),
                             np.percentile(all_val, 40),np.percentile(all_val, 50),np.percentile(all_val, 60),
                             np.percentile(all_val, 70),np.percentile(all_val, 80),np.percentile(all_val, 90)])
            for feature_val in val:
                x1, x2, y1, y2 = self.split(X, y, feature_idx, feature_val)
                current_split = {'feature_idx': feature_idx, 'feature_val': feature_val, 
                                'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'best_score': best_score}
                score = self.get_cost(y1, y2)
                if score < best_score:
                    flag = True
                    best_score = score
                    best_split = current_split
        if flag:
            return best_split['feature_idx'], best_split['feature_val'], best_split['best_score'], \
                    best_split['x1'], best_split['x2'], best_split['y1'], best_split['y2']
        else:
            return current_split['feature_idx'], current_split['feature_val'], current_split['best_score'], \
                    current_split['x1'], current_split['x2'], current_split['y1'], current_split['y2']
    

    def fit(self, X: np.array, y: np.array, current_depth: int = 0, parent = None) -> RegressionTreeNode:
        """Build a decision tree for the given data
        Args:
            X (np.array): The input data
            max_depth (int): The maximum depth of the tree
        Returns:
            root (RegressionTreeNode): The root node of the decision tree"""
        
        self.total_nclasses = len(np.unique(y))
        

        if current_depth > self.max_depth:
            return None
        
        feature_idx, feature_val, best_score, x1, x2, y1, y2 = self.find_best_split(X, y)
        tree = RegressionTreeNode(X, feature_idx, feature_val, parent)
        tree.predict_values = np.mean(y)
        
        if len(x1) < self.min_samples_leaf or len(x2) < self.min_samples_leaf:
            self.tree = tree
            return self.tree
        #print(current_depth)
        current_depth += 1
        tree.left = (self.fit(x1, y1, current_depth, parent = tree))
        tree.right = (self.fit(x2, y2, current_depth, parent = tree))

        self.tree = tree
        return self.tree
    
    def predict(self, X: np.array) -> np.array:
        """Predict the class of the given input data
        Args:
            X (np.array): The input data
        Returns:
            predictions (np.array): The predicted class of the input data"""
        predictions = []
        for x in X:
            node = self.tree
            while node.left is not None and node.right is not None:
                if x[node.feature_idx] < node.feature_val:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.predict_values)
        return np.array(predictions)
    
    def print_tree(self, Node: RegressionTreeNode, depth: int = 0):
        """Print the decision tree
        Args:
            Node (RegressionTreeNode): The root node of the tree
            depth (int): The depth of the tree"""
        if depth == 0:
            Node = self.tree
        if Node is None:
            return
        print(' | '*depth,'-', Node.feature_idx, '<' , Node.predict_values)
        if Node.left is None and Node.right is None:
            print('   '*depth, '| - value :', np.argmax(Node.predict_values))
        self.print_tree(Node.left, depth+1)
        print(' | '*depth,'-', Node.feature_idx, '>=' , Node.predict_values)
        if Node.left is None and Node.right is None:
            print('   '*depth, '| - value: ', np.argmax(Node.predict_values))
        self.print_tree(Node.right, depth+1)

class RandomForestClassifier:
    def __init__(self, n_classifiers: int = 3, data_percentage = 0.5, criterion: str = 'entropy', max_depth: int = 6, min_samples_leaf: int = 1):
        """
        Args:
            n_classifiers (int): number of decision trees
            data_percentage (float): percentage of data to be used to train each tree
            criterion (str): criterion to split the data
            max_depth (int): maximum depth of the tree
            min_samples_leaf (int): minimum number of samples required to be at a leaf node
        """
        self.n_classifiers = n_classifiers
        self.data_percentage = data_percentage
        self.classifiers = [DecisionTreeClassifier(criterion, max_depth, min_samples_leaf) for _ in range(n_classifiers)]
    
    def fit(self, X: np.array, Y: np.array):
        """Build multiple decision trees for the given data
        Args:
            X (np.array): The input data
            Y (np.array): The target labels"""
        data_len = len(X)
        idx = random.choices(list(range(data_len)),k=int(data_len*self.data_percentage))

        for i in range(self.n_classifiers):
            sub_X = X[idx]
            sub_Y = Y[idx]
            self.classifiers[i].fit(sub_X, sub_Y)

    def predict(self, X: np.array) -> np.array:
        """Predict the labels for the given data
        Args:
            X (np.array): The input data
        Returns:
            np.array: The predicted labels"""
        predictions = []
        for x in X:
            p = []
            for c in self.classifiers:
                node = c.tree
                while node.left is not None and node.right is not None:
                    if x[node.feature_idx] < node.feature_val:
                        node = node.left
                    else:
                        node = node.right
                p.append(np.argmax(node.label_probs))
            predictions.append(mode(p))
        return np.array(predictions)
    

class AdaBoostClassifier:
    def __init__(self, n_learners: int = 8, sample_percentage: float = 0.8):
        """
        Args:
            n_learners (int): number of decision trees
            sample_percentage (float): percentage of data to be used to train each tree
        """
        self.n_learners = n_learners
        self.sample_percentage = sample_percentage

    def update_weightage(self):
        """Update the weightage of the samples according to the error rate of the learner"""
        y_pred = self.learner.predict(self.X)
        error = np.sum(self.weights * (y_pred != self.y))
        self.amount_of_say.append(math.log((1 - error) / error)+math.log(self.label_count-1))
        d = math.sqrt((1 - error) / error)
        self.weights[y_pred != self.y] *= d
        self.weights[y_pred == self.y] /= d
        self.weights /= np.sum(self.weights)

    def update_train_data(self):
        """Update the training data according to the wieghtage and sample percentage"""
        n_samples = self.X.shape[0]
        sample_size = int(n_samples * self.sample_percentage)
        idx = np.random.choice(n_samples, size=sample_size, replace=True, p=self.weights)
        return self.X[idx], self.y[idx]
        
    def fit(self, X: np.array, y: np.array):
        """Build the AdaBoost model
        Args:
            X: np.array: feature matrix
            y: np.array: target vector
        """
        self.X = X
        self.y = y
        X_sampled = X
        y_sampled = y
        self.n_samples = len(y)
        self.label_count = len(np.unique(y))
        self.weights = np.ones(self.n_samples) / self.n_samples
        self.learners = []
        self.amount_of_say = []

        for i in range(self.n_learners):
            self.learner = DecisionTreeClassifier(max_depth=1)
            self.learner.fit(X_sampled, y_sampled)
            self.update_weightage()
            X_sampled, y_sampled = self.update_train_data()
            self.learners.append(self.learner)

    def predict(self, X: np.array) -> np.array:
        """Predict the target vector
        Args:
            X: np.array: feature matrix
        Returns:
            np.array: predicted target vector
        """
        pred_scores = np.zeros(shape=(self.n_learners, X.shape[0], self.label_count))
        for idx, learner in enumerate(self.learners):
            pred_prob = learner.get_pred_prob(X)
            pred_scores[idx] = np.array(pred_prob)*self.amount_of_say[idx]
        avg_pred_scores = np.mean(pred_scores, axis=0)
        return np.argmax(avg_pred_scores, axis=1)
    
class GradientBoostingRegressor:
    def __init__(self, epochs: int=100, learning_rate: float=0.1):
        """
        Args:
            epochs (int): number of epochs
            learning_rate (float): learning rate
        """
        self.epochs = epochs
        self.learning_rate = learning_rate

    def calculate_grdient(self, y: np.array, y_pred: np.array) -> np.array:
        """Calculate the gradient of the loss function
        Args:
            y: np.array: target vector
            y_pred: np.array: predicted target vector
        Returns:
            np.array: gradient of the loss function
        """
        return -(y - y_pred)
    
    def calculate_loss(self, y: np.array, y_pred: np.array) -> float:
        """Calculate the loss function
        Args:
            y: np.array: target vector
            y_pred: np.array: predicted target vector
        Returns:
            float: loss value
        """
        return np.sqrt(np.mean((y - y_pred)**2))

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y
        self.models = []
        self.losses = []
        y_pred = np.zeros(len(y)).reshape(-1, 1)*np.mean(y)
        for _ in range(self.epochs):
            if _ + 1 % 10 == 0:
                print("Epoch: ", _, "Loss: ", self.calculate_loss(y, y_pred))
            residual = self.calculate_grdient(y, y_pred)
            model = DecisionTreeRegressor(max_depth=1)
            model.fit(X, residual)
            m_pred = model.predict(X).reshape(-1,1)
            y_pred -= self.learning_rate*m_pred
            self.models.append(model)
            self.losses.append(self.calculate_loss(y, y_pred))
            if _ > 10:
                if (self.losses[-2] - self.losses[-1]) < 0.001:
                    print("Epoch: ", _, "Loss: ", self.calculate_loss(y, y_pred))
                    break

    def predict(self, X: np.array) -> np.array:
        """Predict the target vector
        Args:
            X: np.array: feature matrix
        Returns:
            np.array: predicted target vector
        """
        y_pred = np.zeros(len(X)).reshape(-1, 1)*np.mean(self.y)
        for model in self.models:
            y_pred -= self.learning_rate*model.predict(X).reshape(-1, 1)
        return y_pred

    def plot_loss(self):
        """Plot the loss function"""
        df = pd.DataFrame({'epochs': range(1, len(self.losses)+1), 'losses': self.losses})
        sns.set_theme()
        sns.lineplot(x='epochs', y='losses', data=df)

class LogisticRegression:
    def __init__(self, lr: float=0.01, epochs: int=1000, fit_intercept: bool=False, verbose: bool=True):
        """
        Args:
            lr (float): learning rate
            epochs (int): number of epochs
            fit_intercept (bool): whether to fit an intercept term in addition to the coefficients
            verbose (bool): whether to output loss after each epoch
        """
        self.lr = lr
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def add_intercept(self, X: np.array) -> np.array:
        """Add intercept to the input data
        Args:
            X (np.array): input data
        Returns:
            np.array: input data with intercept
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z: np.array) -> np.array:
        """Calculate the sigmoid function
        Args:
            z (np.array): input data
        Returns:
            np.array: sigmoid of the input data
        """
        return 1 / (1 + np.exp(-z) + 1e-10)
    
    def loss(self, h: np.array, y: np.array) -> float:
        """Calculate the loss function
        Args:
            h (np.array): predicted output of sigmoid function
            y (np.array): ground truth
        Returns:
            float: loss value
        """
        return np.sum(np.log(1 + np.exp(h)) - y * h)
    
    def fit(self, X: np.array, y: np.array):
        """Fit the model
        Args:
            X (np.array): input data
            y (np.array): ground truth
        """
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        self.losses = []

        for i in range(self.epochs):
            z = np.dot(X, self.theta)  ##f(x)=B^TX
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            self.losses.append(self.loss(h, y))

            if(self.verbose == True and i % 100 == 0):
                print(f'loss: {self.loss(h, y)} \t')

    def predict_prob(self, X: np.array) -> np.array:
        """Predict the probability of the input data
        Args:
            X (np.array): input data
        Returns:
            np.array: predicted probability
        """
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X: np.array, threshold: int=0.5) -> np.array:
        """Predict the class of the input data
        Args:
            X (np.array): input data
            threshold (int): threshold for classification
        Returns:
            np.array: predicted class
        """
        return self.predict_prob(X) >= threshold