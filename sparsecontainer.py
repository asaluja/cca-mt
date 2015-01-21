#!/usr/bin/python -tt

'''
File: sparsecontainer.py (formerly eigentypepy)
Date: December 25, 2014 (major changes: January 19, 2015)
Description: contains SparseContainer, SparseContext (child
of SparseContainer), and ContextExtractor classes. 
'''

import sys, commands, string
import numpy as np
import scipy.sparse as sp

'''
base class for eigenphrase and eigencontext classes (below).
Contains in-built OOV handling, with the cut-off as an argument
to the constructor. Types with frequency <= cutoff will be aggregated
into an <unk> token with a parameter based on the aggregated <= cutoff
count types. Eigentype and its child class eigencontext are meant for
feature extraction and should contain sparse token matrices (as well 
as the means to construct them), OOV information, and a dictionary that 
maps between type strings and IDs
'''
class SparseContainer(object):
    def __init__(self, oov_cutoff = 0):
        self.type_id_map = {} #read/write DS
        self._rows = []
        self._cols = []
        self._vals = []
        self._cutoff = oov_cutoff
        self._oov_map = {}
        self._counter = 0
        self.token_matrix = None

    '''
    Note: function below takes a list as an argument and can be used by
    SparseContainer as well as the SparseContext child class
    '''
    def add_token(self, features):
        for feature in features:
            if feature in self.type_id_map: #features make it here only after they exceed the oov cutoff, so we increment as normal
                self._rows.append(self._counter)
                self._cols.append(self.type_id_map[feature])
                self._vals.append(1.)
            else: #either feature is in oov map or this is the first time we've seen this feature
                count, row_idxs = self._oov_map[feature] if feature in self._oov_map else (0, []) #pull up count and row_idxs, else assign init
                count += 1
                row_idxs.append(self._counter)
                if count > self._cutoff: #if cut off exceeded, add to seen features
                    feature_id = len(self.type_id_map) #obtain feature ID
                    self.type_id_map[feature] = feature_id
                    for row_idx in row_idxs:
                        self._rows.append(row_idx)
                        self._cols.append(feature_id)
                        self._vals.append(1.)
                    self._oov_map.pop(feature, None) #and remove from potential OOV features (returns None if not there)
                else: #otherwise, add to OOV features
                    self._oov_map[feature] = (count, row_idxs)
        self._counter += 1

    '''
    creates scipy csr matrix based on collected features, and also handles OOVs
    '''
    def create_sparse_matrix(self):
        if len(self._oov_map) > 0: #there is at least one feature that we have seen <= self.cut_off times
            print "Number of tokens <= oov cut-off %d: %d"%(self._cutoff, len(self._oov_map))
            oov_feat_id = len(self.type_id_map) #corresponds to id of new OOV feat
            self.type_id_map["<unk>"] = oov_feat_id #unk is position-independent                
            for low_freq_token in self._oov_map: #loop through and add indicator to <unk> (pos-dep if flagged)
                count, row_idxs = self._oov_map[low_freq_token]
                for row_idx in row_idxs:
                    self._rows.append(row_idx)
                    self._cols.append(oov_feat_id)
                    self._vals.append(1.)
        self.token_matrix = sp.csr_matrix((self._vals, (self._rows, self._cols)), shape = (self._counter, len(self.type_id_map)))

    def get_type_map(self):
        return self.type_id_map

    def get_token_matrix(self):
        if self.token_matrix is not None:
            return self.token_matrix
        else:
            sys.stderr.write("ERROR! Token matrix has not been estimated or set; returning None\n")
            return None

class SparseContext(SparseContainer):
    def __init__(self, oov_cutoff = 0):
        super(SparseContext, self).__init__(oov_cutoff) #initializing the parent parameters

    def create_sparse_matrix(self, pos_depend, con_length):
        if len(self._oov_map) > 0: #there is at least one feature that we have seen <= self.cut_off times
            print "Number of tokens <= oov cut-off %d: %d"%(self._cutoff, len(self._oov_map))
            oov_feat_id = len(self.type_id_map) #corresponds to id of new OOV feat
            if pos_depend:
                for dist in range(con_length): #add position-dependent OOVs
                    key = "<unk>_dist%d"%(dist+1)
                    self.type_id_map[key] = oov_feat_id
                    oov_feat_id = len(self.type_id_map) #update oov_feat_id for next iteration - otherwise it is unchanged
            else:
                self.type_id_map["<unk>"] = oov_feat_id #unk is position-independent
            for low_freq_token in self._oov_map: #loop through and add indicator to <unk> (pos-dep if flagged)
                count, row_idxs = self._oov_map[low_freq_token]
                if pos_depend: #update oov_feat_id
                    oov_key = "<unk>_dist" + low_freq_token.split('_dist')[1]
                    oov_feat_id = self.type_id_map[oov_key]
                for row_idx in row_idxs:
                    self._rows.append(row_idx)
                    self._cols.append(oov_feat_id)
                    self._vals.append(1.)
        self.token_matrix = sp.csr_matrix((self._vals, (self._rows, self._cols)), shape = (self._counter, len(self.type_id_map)))

    def rescale_features(self, factor):
        var_approx = self.token_matrix.transpose(copy=True)
        var_approx.data **= 2
        scale_denom = var_approx.sum(axis=1) + factor
        scale_vec = np.sqrt((self._counter -1) * np.reciprocal(scale_denom))
        scale_vec_sp = sp.spdiags(scale_vec.flatten(), [0], len(scale_vec), len(scale_vec))
        self.token_matrix = self.token_matrix.dot(scale_vec_sp)
        

class ContextExtractor:
    '''constructor that takes into account context window size, position dependence, and any filtering required'''
    def __init__(self, con_length, pos_depend, stop_words, topN_features):
        self.con_length = con_length
        self.pos_depend = pos_depend
        self.filter_stop = stop_words != ""
        self.stop_words = []
        self.filter_features = topN_features != ""
        self.freq_features = []
        if self.filter_stop: #read in stop words            
            sw_fh = open(stop_words, 'rb')
            for line in sw_fh:
                self.stop_words.append(line.strip())
            sw_fh.close()
        if self.filter_features: #read in most frequent words
            freq_fh = open(topN_features, 'rb')
            for line in freq_fh:
                self.freq_features.append(line.strip())
            freq_fh.close()

    '''
    function that extracts the context of a phrase (given left and right indices of its span)
    from a sentence provided as a list.  The function does the required checks if stop-word 
    filtering is enabled and/or a provided list of the most frequent words is provided, in which
    case we restrict our feature space to those words, and any words outside the list are replaced
    with <OTHER>.  Note that this function may return empty lists for the left/right contexts if
    all con_length context words are stop words. 
    '''
    def extract_context(self, sentence_items, left_idx, right_idx):
        left_con_idx = left_idx - self.con_length
        left_con_words = []
        while left_con_idx < left_idx:
            if left_con_idx < 0:
                left_con_words.append("<s>") #start of sentence marker
                left_con_idx = 0 #skip to beginning of sentence
            else:
                context_word = sentence_items[left_con_idx]
                if not self.filter_stop or not context_word in self.stop_words: #if not checking for sw, or if checking, context is not in sw
                    if self.filter_features and context_word not in self.freq_features: #if checking for listed featureds and context not in list
                        context_word = "<OTHER>" #then replace with other
                    if self.pos_depend: #decorate with distance from word if position dependent
                        context_word += "_dist%d"%(left_idx-left_con_idx)
                    left_con_words.append(context_word) 
                left_con_idx += 1 #skips if its a stop word
        right_con_idx = right_idx + 1
        right_con_words = []
        while right_con_idx < right_idx + self.con_length + 1: #right side; symmetric to left
            if right_con_idx >= len(sentence_items):
                right_con_words.append("</s>")
                break
            else:
                context_word = sentence_items[right_con_idx]
                if not self.filter_stop or not context_word in self.stop_words:
                    if self.filter_features and context_word not in self.freq_features:
                        context_word = "<OTHER>"                        
                    if self.pos_depend:
                        context_word += "_dist%d"%(right_con_idx-right_idx)  
                    right_con_words.append(context_word)
                right_con_idx += 1
        return left_con_words, right_con_words

class config:
    '''constructor reads in config parameters from file'''
    def __init__(self, filename):
        self.file_locs = {}
        fh = open(filename, 'rb')
        for line in fh:
            structure, file_loc = line.strip().split('=')
            self.file_locs[structure] = file_loc

    def get_fileloc(self, eigtype):
        if eigtype in self.file_locs:
            return self.file_locs[eigtype]
        else:
            sys.stderr.write("Eigentype '%s' not found, so cannot read/write. The field name is most likely incorrect\n")
            return ""

    

        


        