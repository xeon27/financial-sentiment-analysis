# all text preprocessing functions

import nltk
import numpy as np
import pandas as pd
import string
import torch

from collections import Counter
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torchtext.vocab import GloVe
from torch.utils.data import TensorDataset
from transformers import BertTokenizer


nltk.download('stopwords')
STOP_WORDS = stopwords.words('english')
PUNCTUATIONS = list(string.punctuation)


class CustomVocab():
    def __init__(self, counter, unk_token='<unk>', pad_token='<pad>'):
        self.words = [unk_token, pad_token] + list(counter.keys())
        self.words_dict = dict(zip(self.words, range(len(self.words))))
        self.vocab_vectors = None
        self.unk_index = self.words.index(unk_token)

    def stoi(self, word):
        return self.words_dict.get(word, self.unk_index)

    def itos(self, idx):
        return self.words[idx]

    def vectorize(self, type):
        if type=='glove':
            embedding_glove = GloVe(name='6B', dim=100)

        self.vocab_vectors = torch.cat([embedding_glove[x].unsqueeze(0) for x in self.words], axis = 0)


class DataPreProcess():
    def __init__(self, test_frac, random_state=None):
        super().__init__()

        self.test_frac = test_frac
        self.random_state = random_state

        self.raw_X_test = None

        self.Y_enc = None
        self.vocab_vectors = None
        self.tfidf_feature_names = None
        self.bert_tokenizer = None
 
    def read_data(self, filename):
        '''
        Reads data from a file into a dataframe
        '''
        data_df = pd.read_csv(filename, header=None, delimiter="@", encoding='latin-1')
        data_df.columns = ["Sentence", "Sentiment"]

        return np.array(data_df["Sentence"]), np.array(data_df["Sentiment"])

    def sentence_tokenize(sentences):
        data = []
        for i in sent_tokenize(sentences):
            temp = []
            for j in word_tokenize(i):
                temp.append(j.lower())
        data.append(temp)
        return data

    def preprocess_X(self):
        pass

    def text_to_vectors(self, X_train, X_test, method, sent_embed=False):
        
        if method=='tfidf':
            tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=STOP_WORDS, min_df=0.005)
            # tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=STOP_WORDS)
            X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
            X_test = tfidf_vectorizer.transform(X_test).toarray()
            self.tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        if method=='sentence_bert':
            X_train = self.get_bert_embedding(X_train)
            X_test = self.get_bert_embedding(X_test)
    
        if method=='glove':
            df = pd.DataFrame({'text': X_train})
            df['text'] = df['text'].apply(lambda x: self.preprocess_sent(x))

            token_dict = Counter(sum(df['text'].values, []))
            my_vocab = CustomVocab(token_dict)
            my_vocab.vectorize(type='glove')
            train_padded = self.pad_text(df['text'].values)
            df = pd.DataFrame({'text': train_padded})
            df['text'] = df['text'].apply(lambda x: np.expand_dims(np.array([my_vocab.stoi(token) for token in x]), 0))
            X_train = torch.from_numpy(np.concatenate(df['text'].values, axis=0))

            df = pd.DataFrame({'text': X_test})
            df['text'] = df['text'].apply(lambda x: self.preprocess_sent(x))
            test_padded = self.pad_text(df['text'].values)
            df = pd.DataFrame({'text': test_padded})
            df['text'] = df['text'].apply(lambda x: np.expand_dims(np.array([my_vocab.stoi(token) for token in x]), 0))
            X_test = torch.from_numpy(np.concatenate(df['text'].values, axis=0))

            self.vocab_vectors = my_vocab.vocab_vectors

        return X_train, X_test

    def _encode_Y(self, Y, one_hot=False):
        if one_hot:
            self.Y_enc = OneHotEncoder()
            Y = Y.reshape(-1, 1)
        else:
            self.Y_enc = LabelEncoder()

        Y_encoded = self.Y_enc.fit_transform(Y)
        
        return Y_encoded.toarray() if one_hot else Y_encoded

    def decode_Y(self, Y):

        Y_decoded = self.Y_enc.inverse_transform(Y)
        
        return Y_decoded

    def prepare_data(self, filename, method, label_one_hot=True):

        X, Y = self.read_data(filename)
        Y = self._encode_Y(Y, one_hot=label_one_hot)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_frac, stratify=Y, 
            random_state=self.random_state)

        self.raw_X_test = X_test

        if method == 'bert':
            X_train_vects, X_test_vects = self.prepare_bert_input(
                X_train, X_test, Y_train, Y_test)
        else:
            X_train_vects, X_test_vects = self.text_to_vectors(
                X_train, X_test, method=method)

        return X_train_vects, X_test_vects, Y_train, Y_test

    def prepare_bert_input(self, X_train, X_test, Y_train, Y_test, method_type='bert-base-uncased'):
        tokenizer = BertTokenizer.from_pretrained(method_type, do_lower_case=True)
        max_len = max([len(tokenizer.encode(sent, add_special_tokens=True)) for sent in X_train])

        train_input_ids = []
        train_attention_masks = []
        for sent in X_train:
            encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True, 
                            max_length = max_len, 
                            pad_to_max_length = True, 
                            return_attention_mask = True, 
                            return_tensors = 'pt',
                            )
            train_input_ids.append(encoded_dict['input_ids'])
            train_attention_masks.append(encoded_dict['attention_mask'])
        
        train_input_ids = torch.cat(train_input_ids, dim=0)
        train_attention_masks = torch.cat(train_attention_masks, dim=0)
        train_labels = torch.tensor(Y_train)

        test_input_ids = []
        test_attention_masks = []
        for sent in X_test:
            encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True, 
                            max_length = max_len, 
                            pad_to_max_length = True, 
                            return_attention_mask = True, 
                            return_tensors = 'pt',
                            )
            test_input_ids.append(encoded_dict['input_ids'])
            test_attention_masks.append(encoded_dict['attention_mask'])
        
        test_input_ids = torch.cat(test_input_ids, dim=0)
        test_attention_masks = torch.cat(test_attention_masks, dim=0)
        test_labels = torch.tensor(Y_test)

        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        
        self.bert_tokenizer = tokenizer

        return train_dataset, test_dataset

    def get_bert_embedding(self, sentences):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(sentences)
        return embeddings

    def preprocess_sent(self, input_sent, remove_sw=True, remove_punct=True):
        tokenized_sent = [token.lower() for token in input_sent.split(" ")]
        tokens_to_remove = [] + (list(STOP_WORDS) if remove_sw else []) + (list(PUNCTUATIONS) if remove_punct else [])
        if remove_sw or remove_punct:
            tokenized_sent = [token for token in tokenized_sent if token not in tokens_to_remove]

        return tokenized_sent

    def pad_text(self, sent_list, pad_token='<pad>'):
        max_length = max([len(sent) for sent in sent_list])
        
        return [(sent+[pad_token]*(max_length-len(sent))) for sent in sent_list]