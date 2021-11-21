from collections import Counter
import pandas as pd
import numpy as np


class BOW():

    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame):

        self.text = df_train.new_keyword.tolist() + df_test.new_keyword.tolist()
        self.bow = dict()
        self.tokens = list()

    def create_bow(self):
        token_counts = Counter()

        for text in self.text:
            token_counts.update(text.split())

        self.tokens = sorted(t for t, c in token_counts.items())

        self.tokens = ['UNK', 'PAD'] + self.tokens

        self.bow = {t: i for i, t in enumerate(self.tokens)}


    def get_len_tokes(self):

        if self.tokens == 0:
            self.create_bow()
            return len(self.tokens)
        else:
            return len(self.tokens)

    def get_matrix(self,
                   lines: list) -> np.array:

        UNK_IX, PAD_IX = self.bow['UNK'], self.bow['PAD']
        seq = [i.split() for i in lines]
        max_len = max(map(len, seq))

        matrix = np.full((len(seq), max_len), PAD_IX)

        for i, s in enumerate(seq):
            row_ix = [self.bow.get(word, UNK_IX) for word in s[:max_len]]
            matrix[i, :len(row_ix)] = row_ix
        return matrix
