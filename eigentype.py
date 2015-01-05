#!/usr/bin/python -tt

'''
'''

import sys, commands, string, gzip
import numpy as np
import scipy.sparse as sp

class eigentype:
    def __init__(self):
        self.type_id_map = {}
        self._rows = []
        self._cols = []
        self._vals = []
        self._counter = 0

    def add_token(self, features):
        for feature in features:
            self._rows.append(self._counter)
            feature_id = self.type_id_map[feature] if feature in self.type_id_map else len(self.type_id_map)
            if feature not in self.type_id_map:
                self.type_id_map[feature] = feature_id
            self._cols.append(feature_id)
            self._vals.append(1.)
        self._counter += 1

    def create_sparse_matrix(self):
        self.token_mat = sp.csc_matrix((self._vals, (self._rows, self._cols)), shape = (self._counter, len(self.type_id_map)))

    def project(self, input_mat):
        return input_mat.dot(self.projection_mat)

    def get_representation(self, features):
        rowIDs = []
        colIDs = []
        vals = []
        for feature in features:
            if feature in self.type_id_map:
                rowIDs.append(0)        
                colIDs.append(self.type_id_map[feature])
                vals.append(1.)
        if len(rowIDs) > 0:
            one_hot = sp.csr_matrix((vals, (rowIDs, colIDs)), shape = (1, len(self.type_id_map)))
            return self.project(one_hot)
        else:
            return None

class context_extractor:
    '''empty initializer'''
    def __init__(self, con_length, pos_depend):
        self.con_length = con_length
        self.pos_depend = pos_depend

    def extract_context(self, sentence_items, left_idx, right_idx):
        left_con_idx = left_idx - self.con_length
        left_con_words = []
        while left_con_idx < left_idx:
            if left_con_idx < 0:
                left_con_words.append("<s>")
                left_con_idx = 0
            else:
                context_word = sentence_items[left_con_idx]
                if self.pos_depend:
                    context_word += "_dist%d"%(left_idx-left_con_idx)
                left_con_words.append(context_word) #may want to decorate word with distance to distinguish
                left_con_idx += 1
        right_con_idx = right_idx + 1
        right_con_words = []
        while right_con_idx < right_idx + self.con_length + 1:
            if right_con_idx >= len(sentence_items):
                right_con_words.append("</s>")
                break
            else:
                context_word = sentence_items[right_con_idx]
                if self.pos_depend:
                    context_word += "_dist%d"%(right_con_idx-right_idx)
                right_con_words.append(context_word)
                right_con_idx += 1
        return left_con_words, right_con_words


        
