# Main file
import argparse
import numpy as np
import os
import pandas as pd

from sklearn.metrics import classification_report
from utils.model_utils import Classifier
from utils.preprocess_utils import DataPreProcess
from utils.shap_utils import ShapInterpreter


def main(args):

    MODEL = args.model.lower()
    TEXT_EMBEDDING = args.embedding.lower()
    TUNE = True if (args.tune).lower()=='t' else False

    random_state = 37
    data_folder = 'data/'
    use_file = "Sentences_50Agree.txt"

    one_hot = False
    if MODEL == 'cnn':
        one_hot = True

    dataProcessor = DataPreProcess(test_frac=0.2, random_state=random_state)
    X_train, X_test, Y_train, Y_test = dataProcessor.prepare_data(data_folder + use_file, TEXT_EMBEDDING, label_one_hot=one_hot)
    if MODEL != 'bert':
        print(X_train.shape, X_test.shape)

    clf = Classifier(model_name=MODEL, random_state=random_state)
    clf.fit(X_train, Y_train, X_test=X_test, word_embeddings=dataProcessor.vocab_vectors, tune=TUNE, save=True)

    y_pred = clf.predict(X_test)
    if one_hot:
        y_pred = np.array([dataProcessor.Y_enc.categories_[0][idx] for idx in y_pred])
        Y_test = dataProcessor.decode_Y(Y_test)

    # print(clf.model_params)
    if one_hot:
        print(classification_report(Y_test, y_pred))
    else:
        print(classification_report(Y_test, y_pred, target_names=dataProcessor.Y_enc.classes_))

    # store test predictions
    if os.path.isfile('results/test_predictions.csv'):
        df = pd.read_csv('results/test_predictions.csv')
    else:
        df = pd.DataFrame({"sentences": dataProcessor.raw_X_test, "sentiment": dataProcessor.decode_Y(Y_test)})
    df[f"{MODEL} predictions"] = [dataProcessor.Y_enc.classes_[idx] for idx in y_pred]
    print(df.shape)
    df.to_csv('results/test_predictions.csv', index=False)

    # shap plots
    clf_shap = ShapInterpreter(
        MODEL, X_train, X_test, label_map=dataProcessor.Y_enc.classes_, 
        feature_names=dataProcessor.tfidf_feature_names, bert_tokenizer=dataProcessor.bert_tokenizer)
    clf_shap.summary(X_test)

    # sample_indices = []
    # mask = (df['sentiment']!=df['xgb predictions']) & (df['sentiment']==df['bert predictions'])
    # for label in ['negative', 'neutral', 'positive']:
    #     mask_1 = mask & (df['sentiment']==label)
    #     sample_indices.append(df[mask_1].sample(1, random_state=53).index[0])
    # print(sample_indices)

    # for sample_idx in sample_indices:
    #     print(f"Sentence [{sample_idx}]: {df['sentences'].iloc[sample_idx]}")
    #     print(f"Actual: {df['sentiment'].iloc[sample_idx]}\nXGB: {df['xgb predictions'].iloc[sample_idx]}\nBERT: {df['bert predictions'].iloc[sample_idx]}\n")
    #     clf_shap.force_plot(
    #         sample_idx=sample_idx, y_true=df['sentiment'], 
    #         y_pred=df['xgb predictions'], X=X_test)

    df = pd.DataFrame({"true": Y_test, "pred": y_pred})
    sample_idx = df[(df['true']==2) & (df['pred']==2)].index[1]
    print(dataProcessor.raw_X_test[sample_idx])
    clf_shap.force_plot(sample_idx=sample_idx, y_true=Y_test, y_pred=y_pred, X=dataProcessor.raw_X_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="ML model type")
    parser.add_argument("--embedding", help="Text embedding type", default='tfidf', required=False)
    parser.add_argument("--tune", help="Whether to tune or load existing model", default='f', required=False)

    args = parser.parse_args()

    main(args)