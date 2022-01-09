# all model classes

import joblib
import numpy as np
import os
import random
import time
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


MODEL_DIR = 'models/'
MODEL_DIR_MAP = {
    "lr": 'logistic_regression',
    "svm": 'svm',
    "rf": 'random_forest',
    "xgb": 'xgboost',
    "bert": 'bert',
    "cnn": '1d_cnn'
}


class Classifier():

    def __init__(self, model_name, random_state):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.model_params = None
        self.random_state = random_state
        self.batch_size = None

    def fit(self, X_train, Y_train, X_test=None, word_embeddings=None, tune=False, save=True):
        if self.model_name not in ['bert', 'cnn']:
            model_file = os.path.join(MODEL_DIR + MODEL_DIR_MAP[self.model_name], 'tfidf_0.005/clf.joblib')
        else:
            model_file = os.path.join(MODEL_DIR + MODEL_DIR_MAP[self.model_name], 'saved_model')

        if self.model_name == 'lr':
            if tune:
                lr = LogisticRegression(random_state=self.random_state)
                parameters = {
                    "penalty": ['l2'], 
                    "C": [0.5, 0.7, 0.8, 1.0], 
                    }

                clf = GridSearchCV(lr, parameters, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)
                clf.fit(X_train, Y_train)
                if save:
                    joblib.dump(clf, model_file)

            else:
                clf = joblib.load(model_file)
            
            self.model = clf.best_estimator_
            self.model_params = clf.best_params_
        
        elif self.model_name == 'rf':
            if tune:
                rf = RandomForestClassifier(random_state=self.random_state)
                parameters = {
                    "n_estimators": [100, 200, 500], 
                    "max_depth": [5, 7, 11, 17, 19, 23], 
                    "max_features": ['auto', 'sqrt', 'log2'],
                    "criterion": ['gini', 'entropy']
                    }

                clf = GridSearchCV(rf, parameters, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)
                clf.fit(X_train, Y_train)
                if save:
                    joblib.dump(clf, model_file)

            else:
                clf = joblib.load(model_file)

            self.model = clf.best_estimator_
            self.model_params = clf.best_params_
        
        elif self.model_name == 'svm':
            if tune:
                svc = SVC(random_state=self.random_state)
                parameters = {
                    "C": [0.5, 0.8, 1.0], 
                    "kernel": ['poly', 'rbf', 'sigmoid'], 
                    "degree": [3, 5, 7, 9],
                    "gamma": ['scale', 'auto']
                    }

                clf = GridSearchCV(svc, parameters, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)
                clf.fit(X_train, Y_train)
                if save:
                    joblib.dump(clf, model_file)

            else:
                clf = joblib.load(model_file)
            
            self.model = clf.best_estimator_
            self.model_params = clf.best_params_
        
        elif self.model_name == 'xgb':
            if tune:
                xgb_classifier = XGBClassifier(random_state=self.random_state, n_jobs=-1, use_label_encoder=False)
                parameters = {
                    "n_estimators": [100, 200, 500], 
                    "max_depth": [5, 9, 13, 23, 47],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.5, 0.8, 1.0],
                }

                clf = GridSearchCV(xgb_classifier, parameters, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2)
                clf.fit(X_train, Y_train)
                if save:
                    joblib.dump(clf, model_file)

            else:
                clf = joblib.load(model_file)

            self.model = clf.best_estimator_
            self.model_params = clf.best_params_

        elif self.model_name == 'cnn':
            model_file = os.path.join(model_file, 'model.pt')
            self.model = Classifier_1D_CNN(
                    embedding=word_embeddings, 
                    dropout_size=0.2, 
                    n_grams=[2, 3, 4], 
                    out_channels=[50, 50, 50], 
                    output_size_dense_layer=10, 
                    classes=3, 
                    num_epochs=30, lr=0.001, batch_size=32)
            if tune:
                self.model.fit(X_train, Y_train)

                if save:
                    torch.save(self.model.state_dict(), model_file)
            else:
                self.model.load_state_dict(torch.load(model_file))

        elif self.model_name == 'bert':
            self.model = CustomBERT()
            self.batch_size = 32
            if tune:
                self.model.finetune(
                    train_dataset=X_train, test_dataset=X_test, num_epochs=4, batch_size=self.batch_size, random_state=self.random_state)

                if save:
                    self.model.save(model_file)
            else:
                self.model.load(model_file)

        else:
            raise ValueError(f'Unsupported model type: {self.model_name}')

    def predict(self, X_test):

        if self.model_name != 'bert':
            y_pred = self.model.predict(X_test)
        else:
            y_pred = self.model.predict(X_test, batch_size=self.batch_size)

        return y_pred

    def interpret_results(self):
        pass

    def classification_results(self, true, pred, pred_prob=None, result_label='test'):

        result_dict = {"acc": (true==pred).mean(), 
                    "precision": precision_score(true, pred, average='macro'), 
                    "recall": recall_score(true, pred, average='macro'), 
                    "f1": f1_score(true, pred, average='macro')}

        if pred_prob is not None:
            result_dict.update({"roc_auc": get_roc_auc(true, pred_prob)})

        return pd.DataFrame(result_dict, index=[result_label]).T


class Classifier_1D_CNN(nn.Module):

    def __init__(self, embedding, n_grams, out_channels, output_size_dense_layer, classes, batch_size, lr, num_epochs, dropout_size):

        super(Classifier_1D_CNN, self).__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.vocab_size, self.embed_dim = embedding.shape
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.out_channels = out_channels
        self.n_grams = n_grams
        self.conv1d_list = nn.ModuleList([
                                    nn.Conv1d(
                                        in_channels=self.embed_dim, 
                                        out_channels=out_channels[i], 
                                        kernel_size=n_grams[i])
                                    for i in range(len(self.n_grams))
                                    ]
                                    )
        
        self.dropout = nn.Dropout(dropout_size)
        self.dense1 = nn.Linear(np.sum(out_channels), output_size_dense_layer)
        self.output_dense = nn.Linear(output_size_dense_layer, classes)
    
    def forward(self, x):

        x = self.embedding(x).float().permute(0,2,1)
        x_conv_list = [F.relu(layer(x)) for layer in self.conv1d_list]

        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        x = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        
        x = self.dense1(x)
        x = self.dropout(x)
        output = self.output_dense(x)
        output = F.softmax(output)

        return output

    def fit(self, train_data, train_labels):

        dataset = TensorDataset(train_data, torch.Tensor(train_labels))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        self.train()
    
        for epoch in range(self.num_epochs):
            for id_batch, (x_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: training loss: {loss.item()}")
 
    def predict(self, test_data, prob=False):

        self.eval()

        pred = self(test_data).detach()
        
        if not prob:
            pred = torch.max(pred, dim=1)[-1]

        return pred.numpy()


class Classifier_NN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_dims, dropouts):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.all_layer_dims = [self.input_size] + hidden_layer_dims + [self.output_size]
        self.linear_layers = nn.ModuleList([nn.Linear(self.all_layer_dims[i], self.all_layer_dims[i+1]) for i in range(len(self.all_layer_dims)-1)])
        
        self.dropouts = dropouts
        self.classes_ = None

    def forward(self, x):
        for layer in range(len(self.linear_layers)-1):
            x = F.relu(F.dropout(self.linear_layers[layer](x), p=self.dropouts[layer]))
        
        x = self.linear_layers[-1](x)
        x = F.softmax(x)
        
        return x

    def fit(self, train_data, train_labels, batch_size, lr, num_epochs):

        dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
    
        for epoch in range(num_epochs):
            for id_batch, (x_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self(torch.Tensor(x_batch))
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch}: traing loss: {loss.item()}")
 
    def predict(self, test_data, prob=False):
        self.eval()
        
        pred = self(torch.Tensor(test_data)).detach()

        if not prob:
            pred = torch.max(pred, dim=1)[-1]

        return pred.numpy()


class CustomBERT():

    def __init__(self, model_type="bert-base-uncased", to_cuda=True):

        self.model = BertForSequenceClassification.from_pretrained(
            model_type, num_labels=3, output_attentions=False, output_hidden_states=False)
        self.device = torch.device("cpu")
        if to_cuda:
            self.device = torch.device("cuda")
            self.model = self.model.cuda()

    def finetune(self, train_dataset, test_dataset, num_epochs, batch_size, random_state):

        train_loader = DataLoader(
            train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        epochs = num_epochs
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(train_loader):

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                result = self.model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask, 
                    labels=b_labels, return_dict=True)

                loss = result.loss
                logits = result.logits
                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            print("")
            print("Running on Test Data...")
            t0 = time.time()

            test_loader = DataLoader(
                test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            for batch in test_loader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():
                    result = self.model(
                        b_input_ids, token_type_ids=None, attention_mask=b_input_mask, 
                        labels=b_labels, return_dict=True)
                
                loss = result.loss
                logits = result.logits
                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            avg_test_accuracy = total_eval_accuracy / len(test_loader)
            print("  Accuracy: {0:.2f}".format(avg_test_accuracy))

            avg_test_loss = total_eval_loss / len(test_loader)
            test_time = format_time(time.time() - t0)

            print("  Test Loss: {0:.2f}".format(avg_test_loss))
            print("  Run on Test Data took: {:}".format(test_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Test Loss': avg_test_loss,
                    'Test Accuracy': avg_test_accuracy,
                    'Training Time': training_time,
                    'Test Time': test_time
                    }
                    )
        
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    def predict(self, test_dataset, batch_size):

        test_loader = DataLoader(
            test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

        self.model.eval()

        predictions = []
        for batch in test_loader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            
            with torch.no_grad():
                result = self.model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=True)
        
            logits = result.logits
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        
        predictions = np.concatenate(predictions, axis=0)
        predictions = np.argmax(predictions, axis=1).flatten()

        return predictions

    def save(self, file_path):

        self.model.save_pretrained(file_path)

    def load(self, file_path):
        
        self.model = BertForSequenceClassification.from_pretrained(file_path)

        if self.device == torch.device("cuda"):
            self.model = self.model.cuda()

    def flat_accuracy(self, preds, labels):

        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))
