__author__ = "Frank van Mourik"

# General
import numpy as np
import pandas as pd

# Data & training
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# SHAP
from shap import TreeExplainer, LinearExplainer, KernelExplainer, Explainer


class IterSHAP():
    """
    Iterative feature selection based on SHAP
    """

    def __init__(
            self,
            model:any = RandomForestClassifier(),
            max_iter:int = 3,
            step_size:float = 0.50,
            train_split_size:float = 0.60
    ):
        """
        Create an itershap object

        Parameters
        ----------
        model: Any, optional
            Any supported model, used for intermediate and final evaluation of feature selection. Can be a different model than used for classification/regression afterwards
        max_iter: int, optional
            The maximum number of iteration IterSHAP should run
        step_size: float, optional
            The percentage of features IterSHAP should keep after each sub-iteration
        train_split_size: float, optional
            The percentage of input data used for training. Remainder split 50/50 over validation and shap. Default 60/20/20
        """
        self.model = model
        self.max_iter = max_iter
        self.step_size = step_size
        self.train_split_size = train_split_size
    

    def get_explainer(self, clf):
        """If shap has an optimised Explainer type for the classifier provided, return that Explainer, otherwise return the default
        """
        model_type = type(clf)
        if model_type == RandomForestClassifier or model_type == DecisionTreeClassifier or model_type == CatBoostClassifier or model_type == XGBClassifier:
            return TreeExplainer(clf, self.X_shap)
        elif model_type == SVC:
            return KernelExplainer(clf, self.X_shap)
        elif model_type == RidgeClassifier:
            return LinearExplainer(clf, self.X_shap)
        else:
            raise Explainer(clf, self.X_shap)

    
    def get_shap_important_features(self, clf):
        """Returns a pd.series with as indices the feature names and as values the importance
        """
        explainer = self.get_explainer(clf)
        shap_test = explainer(self.X_shap, check_additivity=False)
        shap_df = pd.DataFrame(shap_test.data, columns=shap_test.feature_names, index=self.X_shap.index)
        sorted_importances = shap_df.apply(np.abs).mean().sort_values(ascending=False)

        # Rescale the importance such that it sums back to one
        total_importance = sorted_importances.sum()
        sorted_importances = sorted_importances.divide(total_importance)
        
        return sorted_importances

    
    def select_features(self, nr_features, LOWER_LIMIT):
        """Initialize, train, and evaluate a classifier
        """
        clf = self.model
        clf.fit(self.X_train, self.y_train)
        y_pred_val = clf.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred_val)
        self.results_log[nr_features] = accuracy

        if nr_features <= LOWER_LIMIT+1:
            return self, 0

        # Number of features to select
        nr_features = round((nr_features + LOWER_LIMIT)*self.step_size)

        # Select features for next iteration
        selected_features = self.get_shap_important_features(clf)
        selected_features = selected_features.index.tolist()[:nr_features]
        self.selected_features_log[nr_features] = selected_features

        # Only keep the most-relevant features:
        self.X_train = self.X_train[selected_features]
        self.X_val = self.X_val[selected_features]
        self.X_shap = self.X_shap[selected_features]

        return self, nr_features
    
    
    def get_limits(self, CURR_BEST_NR_FEATURES, START_NR_FEATURES):
        """Set upper limit to value above best result so far
        """
        if START_NR_FEATURES == CURR_BEST_NR_FEATURES:
            # If the best accuracy was achieved at the highest nr. of features, set the upper limit to the max. nr. of features included after filtering
            UPPER_LIMIT = CURR_BEST_NR_FEATURES
        else:
            # Otherwise, set the upper limit to the nr. of features one setting above the best, which is twice the nr. of features, as we divide by 2 every iteration
            CONFIG_LIST = list(self.selected_features_log.keys())
            BEST_INDEX = CONFIG_LIST.index(CURR_BEST_NR_FEATURES)
            UPPER_LIMIT = CONFIG_LIST[BEST_INDEX+1]

            # This fixed the issue that 25 --> 12, so when multiplying by two, we should check whether in the iteration step, the integer division rounded down or not
            if not UPPER_LIMIT in self.selected_features_log:
                UPPER_LIMIT += 1
        
        # Set lower limit to value below best result so far
        if CURR_BEST_NR_FEATURES<2:
            # If the highest accruacy was achieved at the lowest nr. of features, set the lower limit to 1, as we always need to include at least one feature
            LOWER_LIMIT = 1
        else:
            # Otherwise, set the lower limit to the nr. of features one setting below the best.
            # I.e. when the best nr. of features was 8, we set the lower limit one level lower, so to 5
            CONFIG_LIST = list(self.selected_features_log.keys())
            BEST_INDEX = CONFIG_LIST.index(CURR_BEST_NR_FEATURES)
            LOWER_LIMIT = CONFIG_LIST[BEST_INDEX-1]
        return self, UPPER_LIMIT, LOWER_LIMIT


    def fit(self, X, y):
        """Execute the IterSHAP algorithm and set the best subset of features as property of the class
        """
        X = pd.DataFrame(X)

        # Split the training split. The remaining data is split 50/50 over validation and shap data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, train_size=self.train_split_size)
        self.X_val, self.X_shap, self.y_val, self.y_shap = train_test_split(self.X_val, self.y_val, train_size=0.50)
        self.results_log = {}

        # Keep a copy of the original data to restore after iterations
        self.X_train_origin = self.X_train
        self.X_val_origin = self.X_val
        self.X_shap_origin = self.X_shap

        # Initiate the log with the a list of all features
        START_NR_FEATURES = self.X_train_origin.shape[1]
        self.selected_features_log = {}
        self.selected_features_log[START_NR_FEATURES] = self.X_train.columns.tolist()

        # The initial lower limit of the search space
        LOWER_LIMIT = 0

        # Go down in nr of features with the defiend step_size .
        # At each (sub-)iteration selected the best feature subset
        for _ in range(self.max_iter, 0, -1):
            # The current number of feature in X 
            nr_features = self.X_train.shape[1]

            # Go down the search space until lower limit is reached
            while nr_features > 0:
                self, nr_features = self.select_features(nr_features, LOWER_LIMIT)

            # Select subset with highest model accuracy
            CURR_BEST_NR_FEATURES = sorted(self.results_log, key=self.results_log.get, reverse=True)[0]

            # Order the results_log and selection_features_log
            self.selected_features_log = dict(sorted(self.selected_features_log.items()))
            self.results_log = dict(sorted(self.results_log.items()))

            # Determine the upper and lower limit for the next search space
            self, UPPER_LIMIT, LOWER_LIMIT = self.get_limits(CURR_BEST_NR_FEATURES, START_NR_FEATURES)

            # Break when search space is exhausted
            if UPPER_LIMIT - LOWER_LIMIT <= 2:
                break
            
            # Reinitialize X_train, X_val, and X_shap to upper limit features from the selected features log.
            # This is the start data for the next iteration
            self.X_train = self.X_train_origin[self.selected_features_log[UPPER_LIMIT]]
            self.X_val = self.X_val_origin[self.selected_features_log[UPPER_LIMIT]]
            self.X_shap = self.X_shap_origin[self.selected_features_log[UPPER_LIMIT]]
            
        # Select best performance from the logs.
        BEST_NR_FEATURES = sorted(self.results_log, key=self.results_log.get, reverse=True)[0]
        BEST_SELECTED_FEATURES = self.selected_features_log[BEST_NR_FEATURES]
        
        # Set the best performing subset at property of the class
        self.best_subset = BEST_SELECTED_FEATURES
        return self
    

    def transform(self, X):
        """Transform input X to only include the best features, calculated by IterSHAP. If IterSHAP has not been run yet, return X.
        """
        X = pd.DataFrame(X)
        if self.best_subset:
            return X[self.best_subset]
        return X
