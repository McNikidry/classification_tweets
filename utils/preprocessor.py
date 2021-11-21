import pandas as pd
import numpy as np
import spacy
import tensorflow_hub as hub
import re
from utils import utils
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from types import Union


class Preprocessor:

    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame):

        self.df_train = df_train
        self.df_test = df_test

        self.nlp = spacy.load('en_core_web_sm')

        self.stop_words = stopwords.words('english')

        self.sentence_enc = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    def replace_reduce(self,
                       text: str) -> str:

        lookup_dict = utils.get_lookup_dict()
        words = text.split()
        abbrevs_removed = []

        for i in words:
            if i in lookup_dict:
                i = lookup_dict[i]
            abbrevs_removed.append(i)

        return ' '.join(abbrevs_removed)

    def lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text)
        lemma_sent = [i.lemma_ for i in doc if not i.is_stop]

        return ' '.join(lemma_sent)

    def extract_keywords(self,
                         text: str) -> str:

        potential_keywords = []
        TOP_KEYWORD = -1
        # Create a list for keyword parts of speech
        pos_tag = ['ADJ', 'NOUN', 'PROPN']
        doc = self.nlp(text)
        for i in doc:
            if i.pos_ in pos_tag:
                potential_keywords.append(i.text)

        document_embed = self.sentence_enc([text])
        potential_embed = self.sentence_enc(potential_keywords)

        vector_distances = cosine_similarity(document_embed, potential_embed)
        keyword = [potential_keywords[i] for i in vector_distances.argsort()[0][TOP_KEYWORD:]]

        return keyword

    def keyword_filler(self,
                       keyword: pd.Series,
                       text: pd.Series) -> str:
        if pd.isnull(keyword):
            try:
                keyword = self.extract_keywords(text)[0]
            except:
                keyword = ''

        return keyword

    def drop_duplicate(self):

        self.df_train.drop_duplicates(['text', 'target'], inplace=True, ignore_index=True)
        self.df_train.drop([4253, 4193, 2802, 4554, 4182, 3212, 4249, 4259, 6535, 4319, 4239, 606, 3936, 6018, 5573],
                           inplace=True)
        self.df_train = self.df_train.reset_index(drop=True)

    def fillna_keyword(self):

        self.df_train['new_keyword'] = pd.DataFrame(list(map(self.keyword_filler,
                                                             self.df_train['keyword'],
                                                             self.df_train['text']))).astype(str)

        self.df_test['new_keyword'] = pd.DataFrame(list(map(self.keyword_filler,
                                                            self.df_test['keyword'],
                                                            self.df_test['text']))).astype(str)

    def preprocessing(self) -> Union[pd.DataFrame, pd.DataFrame]:

        self.drop_duplicate()
        self.fillna_keyword()

        puncts = utils.get_punct()
        mispell_dict = utils.get_mispell_dict()

        pattern_new = re.compile(r'\bnew\b')


        ## TRAIN
        self.df_train.text = self.df_train.text.str.replace('\n', ' ')
        # text lower
        self.df_train.text = self.df_train.text.str.lower()
        # replace 2
        self.df_train.text = self.df_train.text.apply(self.replace_reduce)
        # replace 1
        for before, after in mispell_dict.items():
            self.df_train.text = self.df_train.text.str.replace(before, after)
        # text lower
        self.df_train.text = self.df_train.text.str.lower()
        # del url
        self.df_train.text = [re.sub(r'http\S+', '', x) for x in self.df_train.text]
        # del punct
        for punct in puncts:
            self.df_train.text = self.df_train.text.str.replace(punct, '')
        # del digits
        self.df_train.text = [re.sub('\d+', '', line) for line in self.df_train.text]

        self.df_train['text'] = self.df_train['text'].apply(
            lambda x: re.sub(pattern_new, '', x) if pd.isna(x) != True else x)

        ## TEST
        self.df_test.text = self.df_test.text.str.replace('\n', ' ')
        # text lower
        self.df_test.text = self.df_test.text.str.lower()
        # replace 2
        self.df_test.text = self.df_test.text.apply(self.replace_reduce)
        # replace 1
        for before, after in mispell_dict.items():
            self.df_test.text = self.df_test.text.str.replace(before, after)
        # text lower
        self.df_test.text = self.df_test.text.str.lower()
        # del url
        self.df_test.text = [re.sub(r'http\S+', '', x) for x in self.df_test.text]
        # del punct
        for punct in puncts:
            self.df_test.text = self.df_test.text.str.replace(punct, '')
        # del digits
        self.df_test.text = [re.sub('\d+', '', line) for line in self.df_test.text]

        self.df_test['text'] = self.df_test['text'].apply(
            lambda x: re.sub(pattern_new, '', x) if pd.isna(x) != True else x)

        return self.df_train, self.df_test