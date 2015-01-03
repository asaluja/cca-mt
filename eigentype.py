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
        return input_mat*self.projection_mat

        
