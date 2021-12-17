import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators=100, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.forest = []


    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        else:
            self.feature_subsample_size *= X.shape[1]
            self.feature_subsample_size = int(self.feature_subsample_size)

        for i in range(self.n_estimators):
            clf = DecisionTreeRegressor(**self.trees_parameters, max_depth=self.max_depth) # <- needs to fix
            
            train_features = np.arange(X.shape[1])
            train_objects = np.arange(X.shape[0])
            np.random.shuffle(train_features)
            np.random.shuffle(train_objects)
            train_features = train_features[:self.feature_subsample_size]
            low, high = X.shape[0] // 2, X.shape[0]
            train_objects = train_objects[:np.random.randint(low, high)]

            clf.fit(X[train_objects][:, train_features], y[train_objects])
            self.forest.append((clf, train_features))
            if X_val is not None and y_val is not None:
                if i % 10 == 0:
                    rmse = mean_squared_error(y_val, self.predict(X_val))**0.5
                    print(f'iter: {i}\t rmse: {rmse}')



    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predictions = []
        for tree, train_features in self.forest:
            predictions.append(tree.predict(X[:, train_features]))

        return np.mean(np.array(predictions), axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        
        self.forest = []
        self.coefs = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        else:
            self.feature_subsample_size *= X.shape[1]
            self.feature_subsample_size = int(self.feature_subsample_size)

        a = np.zeros(y.shape[0])
        for i in range(self.n_estimators):

            train_features = np.arange(X.shape[1])
            train_objects = np.arange(X.shape[0])
            np.random.shuffle(train_features)
            np.random.shuffle(train_objects)
            train_features = train_features[:self.feature_subsample_size]
            low, high = X.shape[0] // 2, X.shape[0]
            train_objects = train_objects[:np.random.randint(low, high)]

            clf = DecisionTreeRegressor(**self.trees_parameters, max_depth=self.max_depth) # <- needs to fix
            clf.fit(X[train_objects][:, train_features], 2 * (y[train_objects] - a[train_objects]))
            y_pred = clf.predict(X[:, train_features])
            loss = lambda alpha: np.mean(((a + alpha * self.learning_rate * y_pred) - y) ** 2)
            best_alpha = minimize_scalar(loss)
            self.forest.append((clf, train_features))
            self.coefs.append(best_alpha.x * self.learning_rate)
            a += best_alpha.x * self.learning_rate * y_pred

            if X_val is not None and y_val is not None:
                if i % 10 == 0:
                    rmse = mean_squared_error(y_val, self.predict(X_val))**0.5
                    print(f'iter: {i}\t rmse: {rmse}')


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction = np.zeros(X.shape[0])
        for (tree, train_features), coef in zip(self.forest, self.coefs):
            prediction += tree.predict(X[:, train_features]) * coef
        
        return prediction