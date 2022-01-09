import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shap

from transformers import TextClassificationPipeline

from utils.model_utils import CustomBERT


MODEL_DIR = '/w/247/omkardige/csc2515/project/models/'
MODEL_DIR_MAP = {
    "lr": 'logistic_regression',
    "svm": 'svm',
    "rf": 'random_forest',
    "xgb": 'xgboost',
    "bert": 'bert'
}


class ShapInterpreter():

    def __init__(self, model_name, X_train, X_test, label_map, bert_tokenizer=None, feature_names=None):

        self.model_name = model_name
        self.label_map = label_map
        self.feature_names = feature_names
        self.bert_tokenizer = bert_tokenizer
        self.plot_folder = os.path.join(MODEL_DIR + MODEL_DIR_MAP[self.model_name], 'shap_plots')

        self.explainer, self.shap_values = self._get_shap_values(X_train, X_test)

    def _get_shap_values(self, X_train, X_test):
        
        file_name = 'saved_model' if (self.model_name == 'bert') else 'tfidf_all/clf.joblib'
        model_file = os.path.join(MODEL_DIR + MODEL_DIR_MAP[self.model_name], file_name)            

        if self.model_name == 'lr':
            
            clf = joblib.load(model_file)
            model = clf.best_estimator_

            explainer = shap.Explainer(model, X_train, feature_names=self.feature_names)
            shap_values = explainer(X_test)
        
        elif self.model_name == 'xgb':

            clf = joblib.load(model_file)
            model = clf.best_estimator_

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap_values = np.concatenate([np.expand_dims(shap_arr, axis=2) for shap_arr in shap_values], axis=2)
            print(shap_values.shape)

        elif self.model_name == 'bert':

            bert_model = CustomBERT()
            bert_model.load(model_file)
            model = bert_model.model.cpu()

            pipe = TextClassificationPipeline(model=model, tokenizer=self.bert_tokenizer, return_all_scores=True)
            explainer = shap.Explainer(pipe, output_names=self.label_map)
            shap_values = None

        return explainer, shap_values

    def summary(self, X=None, save=True):
        
        for label_idx in range(self.label_map.shape[0]):
            shap.summary_plot(self.shap_values[:,:,label_idx], pd.DataFrame(X, columns=self.feature_names), show=False)
            summary_f = plt.gcf()

            if save:
                summary_f.savefig(
                    os.path.join(self.plot_folder, f'summary_{self.label_map[label_idx]}.png'), 
                    format="png", dpi=150, bbox_inches='tight')
            
            plt.clf()

    def force_plot(self, sample_idx, y_true, y_pred, X=None, save=True):

        if self.model_name == 'bert':
            self.shap_values = self.explainer([X[sample_idx]])

        for label_idx in range(self.label_map.shape[0]):
            if self.model_name == 'bert':
                shap.plots.text(self.shap_values[:, :, label_idx])
            else:
                force_f = shap.force_plot(
                    self.explainer.expected_value[label_idx], self.shap_values[sample_idx, :, label_idx], 
                    pd.DataFrame(X, columns=self.feature_names).round(4).iloc[sample_idx, :], matplotlib=True, show=False)

                # plt.title(
                #     f'True: {self.label_map[y_true[sample_idx]]}; Predicted: {self.label_map[y_pred[sample_idx]]}', 
                #     loc='left')

                if save:
                    force_f.savefig(
                        os.path.join(self.plot_folder, f'force_plot_{sample_idx}_{self.label_map[label_idx]}.png'), 
                            format="png", dpi=150, bbox_inches='tight')

                plt.clf()

