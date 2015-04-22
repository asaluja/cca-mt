#!/usr/bin/python -tt

'''
File: sparsecontainer.py (formerly eigentypepy)
Date: December 25, 2014 (major changes: January 19, 2015)
Description: contains SparseContainer, SparseContext (child
of SparseContainer), and ContextExtractor classes. 
'''

import sys, commands, string, gc
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
    def __init__(self, oov_cutoff, estimate_oov_param):
        self.type_id_map = {} #read/write DS
        self.id_type_map = {} #held-out eval
        self._rows = []
        self._cols = []
        self._vals = []
        self._cutoff = oov_cutoff
        self._oov_map = {}
        self._counter = 0
        self.token_matrix = None
        oov_id = 0 if estimate_oov_param else -1 #OOV assigned ID 0, or -1 if OOV not being used
        self.type_id_map["<unk> ||| <unk>"] = oov_id 
        self.id_type_map[oov_id] = "<unk> ||| <unk>"
        self.estimate_oov = estimate_oov_param

    def add_token(self, token, excluded_tokens):
        if token in self.type_id_map: #features make it here only after they exceed the oov cutoff, so we increment as normal
            self._rows.append(self._counter)
            self._cols.append(self.type_id_map[token])
            self._vals.append(1.)
        else: #either token is in oov map or this is the first time we've seen this token
            count, row_idxs = self._oov_map[token] if token in self._oov_map else (0, []) #pull up count and row_idxs, else assign init
            count += 1 #add current token
            row_idxs.append(self._counter)
            if count > self._cutoff: #if cut off exceeded, add to seen tokens
                token_id = -1
                if token in excluded_tokens: #if phrase pair is in excluded pairs
                    token_id = self.type_id_map["<unk> ||| <unk>"] #either OOV if estimating OOV param or -1
                else: #seen enough times and not in excluded pairs
                    token_id = len(self.type_id_map) if self.estimate_oov else len(self.type_id_map)-1
                    self.type_id_map[token] = token_id #and add to observed features
                    self.id_type_map[token_id] = token
                if token_id > -1: #meeans we are estimating OOV param or the phrase pair is valid and we add it
                    for row_idx in row_idxs:
                        self._rows.append(row_idx)
                        self._cols.append(token_id) 
                        self._vals.append(1.)
                self._oov_map.pop(token, None) #and remove from potential OOV features (returns None if not there)
            else: #otherwise, add to OOV map
                self._oov_map[token] = (count, row_idxs)
        self._counter += 1

    '''
    creates scipy csr matrix based on collected features, and also handles OOVs
    To do: should we filter zero_rows here and then return? 
    '''
    def create_sparse_matrix(self):
        if self.estimate_oov:
            if len(self._oov_map) > 0: #there is at least one feature that we have seen <= self.cut_off times
                print "Number of phrase pair types <= oov cut-off %d: %d"%(self._cutoff, len(self._oov_map))
                oov_feat_id = self.type_id_map["<unk> ||| <unk>"]                
                for low_freq_token in self._oov_map: #loop through and add indicator to <unk> (pos-dep if flagged)
                    count, row_idxs = self._oov_map[low_freq_token]
                    for row_idx in row_idxs:
                        self._rows.append(row_idx)
                        self._cols.append(oov_feat_id)
                        self._vals.append(1.)
            self.token_matrix = sp.csr_matrix((self._vals, (self._rows, self._cols)), shape = (self._counter, len(self.type_id_map)))
        else: #don't want to include OOV entry in token matrix
            self.token_matrix = sp.csr_matrix((self._vals, (self._rows, self._cols)), shape = (self._counter, len(self.type_id_map)-1))

    def filter_zero_rows(self):
        assert self.token_matrix is not None
        rows = []
        cols = []
        vals = []
        counter = 0
        zero_rows = []
        for row_idx in xrange(self.token_matrix.shape[0]):
            dummy, row_cols = self.token_matrix[row_idx,:].nonzero()
            if len(row_cols) > 0: #not a zero row
                for row_col in row_cols:
                    rows.append(counter)
                    cols.append(row_col)
                    vals.append(1.)
                counter += 1
            else:
                zero_rows.append(row_idx)
        self.token_matrix = sp.csr_matrix((vals, (rows, cols)), shape = (counter, len(self.type_id_map)-1))
        return zero_rows

    def get_type_map(self):
        return self.type_id_map

    def get_token_matrix(self, subset_idxs=None):
        if self.token_matrix is not None:
            if subset_idxs is None:
                return self.token_matrix
            else:
                return self.token_matrix[subset_idxs,:]
        else:
            sys.stderr.write("ERROR! Token matrix has not been estimated or set; returning None\n")
            return None

    def get_token_phrase(self, token_id):
        if token_id in self.id_type_map:
            return self.id_type_map[token_id]
        else:
            sys.stderr.write("ERROR! Provided ID not in map!\n")
            sys.exit()

class SparseContext(SparseContainer):
    def __init__(self, oov_cutoff = 0):
        super(SparseContext, self).__init__(oov_cutoff, True) #initializing the parent parameters
        oov_id = self.type_id_map.pop("<unk> ||| <unk>", None) #SparseContext will handle its own OOVs (when constructing sparse matrix)
        assert oov_id is not None
        self.id_type_map.pop(oov_id, None)

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
                    feature_id = len(self.type_id_map)
                    self.type_id_map[feature] = feature_id #and add to observed features
                    self.id_type_map[feature_id] = feature
                    for row_idx in row_idxs:
                        self._rows.append(row_idx)
                        self._cols.append(feature_id) 
                        self._vals.append(1.)
                    self._oov_map.pop(feature, None) #and remove from potential OOV features (returns None if not there)
                else: #otherwise, add to OOV features
                    self._oov_map[feature] = (count, row_idxs)
        self._counter += 1
    
    '''
    ignores OOVs, etc. 
    '''
    def add_token_vec(self, rep):
        self._vals.append(rep)

    def create_dense_matrix(self): 
        stacked_features = np.vstack(self._vals)
        mean_center_vec = np.divide(stacked_features.sum(axis=0), stacked_features.shape[0])
        self.token_matrix = stacked_features - mean_center_vec
        print "Created dense matrix from word vectors"
        del self._vals
        gc.collect()

    def create_sparse_matrix(self, pos_depend, con_length):
        if len(self._oov_map) > 0: #there is at least one feature that we have seen <= self.cut_off times
            print "Number of context types <= oov cut-off %d: %d"%(self._cutoff, len(self._oov_map))
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
        print "Created sparse matrix from counts"

    def filter_zero_rows(self, zero_rows):
        rows = []
        cols = []
        vals = []
        counter = 0
        zero_rows = set(zero_rows) #testing membership is O(1) in set
        original_size = self.token_matrix.shape[0]
        for row_idx in xrange(original_size):
            if row_idx not in zero_rows:
                dummy, row_cols = self.token_matrix[row_idx,:].nonzero()
                assert len(row_cols) > 0
                for row_col in row_cols:                    
                    rows.append(counter)
                    cols.append(row_col)
                    vals.append(1.)
                counter += 1
        assert counter + len(zero_rows) == original_size
        self.token_matrix = sp.csr_matrix((vals, (rows, cols)), shape = (counter, len(self.type_id_map)))
                

class ContextExtractor:
    '''constructor that takes into account context window size, position dependence, and any filtering required'''
    def __init__(self, con_length, pos_depend, stop_words, topN_features, vecs_filename):
        self.con_length = con_length
        self.pos_depend = pos_depend
        self.vec_dim = 0
        self.rep_dict = None
        if vecs_filename != "": #reads in word vectors if defined
            self.rep_dict = self.read_vectors(vecs_filename)
            self.vec_dim = len(self.rep_dict[self.rep_dict.keys()[0]])
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

    def read_vectors(self, filename):
        fh = open(filename, 'r')
        vecs = {}
        for line in fh:
            if len(line.strip().split()) > 2:
                word = line.strip().split()[0]
                rep = np.array([float(i) for i in line.strip().split()[1:]])
                vec_len = np.linalg.norm(rep)
                vecs[word] = np.divide(rep, vec_len) if vec_len > 0 else np.zeros(len(rep))
        fh.close()
        return vecs

    def is_repvec(self):
        return self.vec_dim > 0

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
                word_to_add = np.zeros((self.vec_dim,)) if self.vec_dim > 0 else "<s>"
                left_con_words.append(word_to_add)
                left_con_idx = 0 #skip to beginning of sentence
            else:
                context_word = sentence_items[left_con_idx]
                if not self.filter_stop or not context_word in self.stop_words: #if not checking for sw, or if checking, context is not in sw
                    if self.filter_features and context_word not in self.freq_features: #if checking for listed featureds and context not in list
                        context_word = "<OTHER>" #then replace with other --> assumption is filter_features and wordvecs cannot be both true
                    if self.pos_depend and self.vec_dim == 0: #decorate with distance from word if position dependent and we're not using wordvecs
                        context_word += "_dist%d"%(left_idx-left_con_idx)
                    if self.vec_dim > 0: #using word vectors
                        rep = self.rep_dict[context_word] if context_word in self.rep_dict else np.zeros((self.vec_dim,))
                        context_word = rep
                    left_con_words.append(context_word) 
                left_con_idx += 1 #skips if its a stop word

        right_con_idx = right_idx + 1
        right_con_words = []
        while right_con_idx < right_idx + self.con_length + 1: #right side; symmetric to left
            if right_con_idx >= len(sentence_items):
                word_to_add = self.rep_dict["</s>"] if self.vec_dim > 0 else "</s>"                    
                right_con_words.append(word_to_add)
                break
            else:
                context_word = sentence_items[right_con_idx]
                if not self.filter_stop or not context_word in self.stop_words:
                    if self.filter_features and context_word not in self.freq_features:
                        context_word = "<OTHER>"                        
                    if self.pos_depend and self.vec_dim == 0:
                        context_word += "_dist%d"%(right_con_idx-right_idx)  
                    if self.vec_dim > 0:
                        rep = self.rep_dict[context_word] if context_word in self.rep_dict else np.zeros((self.vec_dim,))
                        context_word = rep
                    right_con_words.append(context_word)
                right_con_idx += 1

        if self.vec_dim > 0 and self.pos_depend: #concatenate the vectors            
            if len(left_con_words) < self.con_length: #check if we need to zero-pad
                diff = self.con_length - len(left_con_words)
                for dummy in range(diff):
                    left_con_words.append(np.zeros((self.vec_dim,)))
            concat_left = np.hstack(left_con_words)
            if len(right_con_words) < self.con_length: #correspondingly on right
                diff = self.con_length - len(right_con_words)
                for dummy in range(diff):
                    right_con_words.append(np.zeros((self.vec_dim,)))
            concat_right = np.hstack(right_con_words)
            return concat_left, concat_right
        elif self.vec_dim > 0:
            sum_left = np.sum(np.vstack(left_con_words), axis=0)
            sum_right = np.sum(np.vstack(right_con_words), axis=0)
            return sum_left, sum_right
        else:
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

    

        


        
