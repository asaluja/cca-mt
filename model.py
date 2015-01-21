#!/usr/bin/python -tt

import sys, commands, string, os
import numpy as np
import scipy.sparse as sp
import scipy.io as io
from mlp import MLPClassifier

'''
for CCA computations: X - left matrix, Y - right matrix, param - rank
for regression: X - desin matrix, Y - response matrix, param - regularization strength
'''
def matlab_interface(X, Y, option, reg_strength, rank = 50):
    pwd = os.getcwd()
    out_loc = pwd + "/matlab_temp"
    io.savemat(out_loc, {'X': X, 'Y': Y})
    path = os.path.abspath(os.path.dirname(sys.argv[0])) #matlab scripts in the same place as this script
    os.chdir(path)
    if option == "CCA":
        os.system('matlab -nodesktop -nosplash -nojvm -r "cca_wrapper ' + out_loc + " %s %s"%(rank, reg_strength) + '"')
    else: #option == "regression"
        os.system('matlab -nodesktop -nosplash -nojvm -r "regression_wrapper ' + out_loc + " %s"%reg_strength + '"')
    os.chdir(pwd)
    mat_return = io.loadmat(out_loc)
    if option == "CCA":
        return mat_return['A'].newbyteorder('='), mat_return['B'].newbyteorder('='), mat_return['r'].newbyteorder('=')
    else:
        return mat_return['beta'].newbyteorder('=')

class Context:
    def __init__(self, left_context, right_context, gamma, rank, concat=False, con_length = 2):
        phi_l, phi_r, lr_correlations = matlab_interface(left_context.get_token_matrix(), right_context.get_token_matrix(), "CCA", gamma, rank)
        self.left_proj_mat = phi_l
        self.right_proj_mat = phi_r
        self.left_map = left_context.get_type_map()
        self.right_map = right_context.get_type_map()
        self.concat = concat
        self.con_length = con_length
        print "Correlations between left and right context (with rank %d): "%rank
        print lr_correlations

    def compute_lowrank_training_contexts(self, train_mat_left, train_mat_right):
        assert train_mat_left.shape[0] == train_mat_right.shape[0] #assert that the number of samples is the same
        lowrank_mat_left, lowrank_mat_right = None, None
        if self.concat:
            num_samples = train_mat_left.shape[0]
            num_features = self.left_proj_mat.shape[1] #left_proj_mat and right_proj_mat have the same # of cols
            lowrank_mat_left = np.zeros((num_samples, self.con_length*num_features))
            lowrank_mat_right = np.zeros((num_samples, self.con_length*num_features))        
            for rowIdx in xrange(num_samples):
                rows, cols = train_mat_left[rowIdx,:].nonzero()
                counter = 0
                for row,col in zip(rows, cols):
                    lowrank_mat_left[rowIdx, counter*num_features:(counter+1)*num_features] = self.left_proj_mat[col,:]
                    counter += 1
                rows, cols = train_mat_right[rowIdx,:].nonzero()
                counter = 0
                for row,col in zip(rows, cols):
                    lowrank_mat_right[rowIdx, counter*num_features:(counter+1)*num_features] = self.right_proj_mat[col,:]
                    counter += 1
        else:
            lowrank_mat_left = train_mat_left.dot(self.left_proj_mat)
            lowrank_mat_right = train_mat_right.dot(self.right_proj_mat)
        return lowrank_mat_left, lowrank_mat_right

    def __get_representation_side(self, features, type_id_map, proj_mat):
        rowIDs = []
        for feature in features:
            if feature in type_id_map:
                rowIDs.append(type_id_map[feature])
            elif "<unk>" in type_id_map: #OOV parameter defined, and it's position-independent
                rowIDs.append(type_id_map["<unk>"])
            else:
                pos_dep = feature.split('_dist')
                if len(pos_dep) > 1: #this means we have pos-dep features, but doesn't necessarily mean OOV parameters are defined
                    oov_key = "<unk>_dist" + pos_dep[1]
                    if oov_key in type_id_map:
                        rowIDs.append(type_id_map[oov_key])        
        if len(rowIDs) > 0: #valid representation            
            result = [] if self.concat else np.zeros((proj_mat.shape[1],))
            for rowID in rowIDs:
                result = np.concatenate((result, proj_mat[rowID,:]), axis=1) if self.concat else result + proj_mat[rowID,:]
            if self.concat and len(result) < self.con_length*proj_mat.shape[1]: #then we need to zero-pad
                diff = self.con_length*proj_mat.shape[1] - len(result)
                result = np.concatenate((result, np.zeros((diff,))), axis=1)
            #good idea to assert length of result here
            return result
        else:
            return None

    def get_representation(self, features_left, features_right):
        left_context_rep = self.__get_representation_side(features_left, self.left_map, self.left_proj_mat)
        right_context_rep = self.__get_representation_side(features_right, self.right_map, self.right_proj_mat)
        return left_context_rep, right_context_rep

class BaseModel(object):
    def __init__(self, type_map, context):
        self.type_id_map = type_map
        self.parameters = None
        self.context = context
        self.inventory = self.__format_phrase_pairs(type_map.keys())

    def __format_phrase_pairs(self, phrase_pairs):
        phrase_dict = {}
        for phrase_pair in phrase_pairs:
            src, tgt = phrase_pair.split(' ||| ')
            translations = phrase_dict[src] if src in phrase_dict else []
            translations.append(tgt)
            phrase_dict[src] = translations
        return phrase_dict               

    def get_tokenID(self, token):
        if token in self.type_id_map:
            return self.type_id_map[token]
        else:
            return -1

    def get_tokens(self):
        return self.type_id_map.keys()

    def get_representation(self, token):
        if token in self.type_id_map:
            assert self.parameters is not None
            return self.parameters[self.type_id_map[token],:]
        else:
            sys.stderr.write("ERROR! Token %s not in inventory\n"%token)
            return None

    def get_context_representation(self, con_left, con_right):
        return self.context.get_representation(con_left, con_right)

    def get_indexed_scores(self, phrase, vec):
        translations = self.inventory[phrase]
        scored_pairs = []
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            score = vec[self.get_tokenID(phrase_pair)]
            scored_pairs.append((phrase_pair, score))
        return scored_pairs

class CCA(BaseModel):
    def __init__(self, context, type_map):
        super(CCA, self).__init__(type_map, context)
        self.context_parameters = None

    def train(self, left_low_rank, right_low_rank, tokens, gamma, rank):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        phi_s, phi_w, cw_correlations = matlab_interface(training_data, tokens.get_token_matrix(), "CCA", gamma, rank)
        self.context_parameters = phi_s
        self.parameters = phi_w
        print "Two-step CCA complete. Correlations between combined context and tokens (with rank %d): "%rank
        print cw_correlations

    def score(self, context_rep, phrase):
        bidi_con_rep = context_rep.dot(self.context_parameters)
        bidi_con_rep_norm = np.linalg.norm(bidi_con_rep)
        translations = self.inventory[phrase]
        scored_pairs = []
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            representation = self.get_representation(phrase_pair)
            score = representation.dot(bidi_con_rep.transpose()) / (np.linalg.norm(representation)*bidi_con_rep_norm)
            scored_pairs.append((phrase_pair, score))
        return scored_pairs

class GLM(BaseModel):
    def __init__(self, context, tokens_map):
        super(GLM, self).__init__(tokens_map, context)

    def train(self, left_low_rank, right_low_rank, tokens, gamma):
        offset = np.ones((left_low_rank.shape[0], 1))
        training_data = np.concatenate((left_low_rank, right_low_rank, offset), axis=1) #if each low-rank context is p-dim, then this is a 2p+1 dim vec
        self.parameters = matlab_interface(training_data, tokens.get_token_matrix(), "regression", gamma)
        print "Fitted general linear model: %d responses %d predictors, %d samples"%(self.parameters.shape[1], self.parameters.shape[0], left_low_rank.shape[0])

    def score(self, context_rep, phrase): 
        aug_context_rep = np.append(context_rep, np.ones((1,)), axis=1)
        predict_vec = aug_context_rep.dot(self.parameters) #1xh vector
        return self.get_indexed_scores(phrase, predict_vec)

    #to do: write shrinkage part

class MLP(BaseModel):
    def __init__(self, context, tokens_map, rank): #there will be other params for the MLP
        super(GLM, self).__init__(tokens_map, context)
        self.mlp = MLPClassifier(n_hidden=rank, verbose=1) #set batch size
    
    def train(self, left_low_rank, right_low_rank, tokens):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        start = time.clock()
        mlp.fit(training_data, tokens.get_token_matrix())
        print "MLP learning complete; input layer size: %d; hidden layer size: %d; output layer size: %d; time taken: %.1f sec"%(training_data.shape[1], self.mlp.n_hidden, tokens.get_token_matrix().shape[1], time.clock()-start)
        
    def score(self, context_rep, phrase):
        predict_vec = self.mlp.predict(context_rep)
        return self.get_indexed_scores(phrase, predict_vec)



        
