#!/usr/bin/python -tt

import sys, commands, string, os, time
import numpy as np
import scipy.sparse as sp
import scipy.io as io
from scipy.special import expit
from mlp import MLPClassifier

def compute_regression(X, Y, reg_strength):
    pwd = os.getcwd()
    out_loc = pwd + "/matlab_temp"
    io.savemat(out_loc, {'X': X, 'Y': Y})
    path = os.path.abspath(os.path.dirname(sys.argv[0])) #matlab scripts in the same place as this script
    os.chdir(path+"/matlab")
    os.system('matlab -nodesktop -nosplash -nojvm -r "regression_wrapper ' + out_loc + " %s"%reg_strength + '"')
    os.chdir(pwd)
    mat_return = io.loadmat(out_loc)
    return mat_return['beta']

def compute_cca(X, Y, reg_strength, rank, approx="full", mean_center=False):
    p1 = X.shape[1]
    p2 = Y.shape[1]
    n = X.shape[0]
    X = sp.csr_matrix(X)
    cross_cov = X.transpose().dot(Y)
    if mean_center:
        feature_counts_left = np.multiply(X.sum(axis=0), 1./n)
        feature_counts_right = np.multiply(Y.sum(axis=0), 1./n)
        mean_center_mat = feature_counts_left.transpose().dot(feature_counts_right)
        mean_center_mat = np.multiply(mean_center_mat, n)
        cross_cov = cross_cov - sp.csc_matrix(mean_center_mat) #dense matrix, but in sparse data structure        
    pwd = os.getcwd()
    out_loc = pwd + "/matlab_temp"
    whiten_left = None
    whiten_right = None
    if approx == "diag": #compute diagonal approx to inverse square root matrix
        X_cp = X
        Y_cp = Y
        X_cp.data **= 2
        Y_cp.data **= 2
        feature_counts_left = X_cp.sum(axis=0) + reg_strength
        feature_counts_right = Y_cp.sum(axis=0) + reg_strength
        scale_vec_left = np.reciprocal(np.sqrt(feature_counts_left))
        scale_vec_right = np.reciprocal(np.sqrt(feature_counts_right))
        whiten_left = sp.spdiags(scale_vec_left.flatten(), [0], p1, p1)
        whiten_right = sp.spdiags(scale_vec_right.flatten(), [0], p2, p2)
        cross_cov = whiten_left*cross_cov*whiten_right #scipy sparse matrices * operator is matrix mult
    elif approx == "ppmi": 
        inv_feat_counts_left = np.reciprocal(X.sum(axis=0)+1) #add one if not in training but only in held-out
        inv_feat_counts_left_scaled = np.multiply(inv_feat_counts_left, n)
        left_mult = sp.spdiags(inv_feat_counts_left_scaled, [0], p1, p1)
        inv_feat_counts_right = np.reciprocal(Y.sum(axis=0)+1)
        right_mult = sp.spdiags(inv_feat_counts_right.flatten(), [0], p2, p2)
        cross_cov = left_mult*cross_cov*right_mult #p(x,y) / (p(x) * p(y))
        cross_cov.data = np.log(cross_cov.data) - np.log(reg_strength) #element-wise log and shift
        cross_cov.data *= cross_cov.data > 0 #filtering out non-zeros
        cross_cov.eliminate_zeros()
    if approx == "full":
        io.savemat(out_loc, {'X': X, 'Y': Y})
        path = os.path.abspath(os.path.dirname(sys.argv[0])) 
        os.chdir(path+"/matlab") #matlab scripts in 'matlab' sub-directory
        os.system('matlab -nodesktop -nosplash -nojvm -r "cca_wrapper ' + out_loc + " %s %s"%(rank, reg_strength) + '"')        
    else:
        io.savemat(out_loc, {'avgOP': cross_cov})
        path = os.path.abspath(os.path.dirname(sys.argv[0])) 
        os.chdir(path+"/matlab")
        os.system('matlab -nodesktop -nosplash -nojvm -r "matlab_svd ' + out_loc + " %s"%rank + '"')
    os.chdir(pwd)
    mat_return = io.loadmat(out_loc)
    if approx == "diag": #if diag approx, multiply result by scaling
        return whiten_left.dot(mat_return['U']), whiten_right.dot(mat_return['V']), mat_return['S']
    else: #for full CCA, already scaled in cca_direct.m
        return mat_return['U'], mat_return['V'], mat_return['S']         

class Context:
    def __init__(self, left_context, right_context, gamma, rank, concat=False, con_length = 2, train_idxs=None, approx="full", mean_center = False):
        phi_l, phi_r, lr_correlations = compute_cca(left_context.get_token_matrix(train_idxs), right_context.get_token_matrix(train_idxs), gamma, rank, approx, mean_center)
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
        else: #guaranteed that if word vectors are active concat is not active
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
            return result
        else:
            return None

    def get_representation(self, features_left, features_right):
        left_context_rep = self.__get_representation_side(features_left, self.left_map, self.left_proj_mat)
        right_context_rep = self.__get_representation_side(features_right, self.right_map, self.right_proj_mat)
        return left_context_rep, right_context_rep

    def get_rep_vec(self, vec_left, vec_right):
        left_context_rep = vec_left.dot(self.left_proj_mat)
        right_context_rep = vec_right.dot(self.right_proj_mat)
        return left_context_rep, right_context_rep

class BaseModel(object):
    '''type_id_map maps phrase pairs for which we have parameters to their indices; inventory contains all phrase pairs (incl. singletons) '''
    def __init__(self, type_map, all_pp, context):
        self.type_id_map = type_map
        self.parameters = None
        self.context = context
        self.inventory = self.__format_phrase_pairs(list(all_pp))

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
            #sys.stderr.write("WARNING! No parameters for token %s - singleton\n"%token)
            return -1

    '''function returns all source phrases in training, including singletons'''
    def get_tokens(self): #TO DO: store singletons as well, return both singletons and other keys
        return self.inventory.keys()
        #return self.type_id_map.keys()

    def get_representation(self, token):
        if token in self.type_id_map:
            assert self.parameters is not None
            return self.parameters[self.type_id_map[token],:]
        else:
            #sys.stderr.write("WARNING! No parameters for token %s - singleton\n"%token)
            return None

    def get_context_rep(self, con_left, con_right):
        return self.context.get_representation(con_left, con_right)

    def get_context_rep_vec(self, vec_left, vec_right):
        return self.context.get_rep_vec(vec_left, vec_right)

    def get_candidate_indices(self, phrase):
        translations = self.inventory[phrase]
        idxs = []
        phrase_pairs = []
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            idxs.append(self.get_tokenID(phrase_pair)) #if phrase pair not in model, appends -1
            phrase_pairs.append(phrase_pair)
        return idxs, phrase_pairs

class CCA(BaseModel):
    def __init__(self, context, type_map, all_pp):
        super(CCA, self).__init__(type_map, all_pp, context)
        self.context_parameters = None

    def train(self, left_low_rank, right_low_rank, tokens, gamma, rank, train_idxs=None, approx="full", mean_center=False):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        phi_s, phi_w, cw_correlations = compute_cca(training_data, tokens.get_token_matrix(train_idxs), gamma, rank, approx, mean_center)
        self.context_parameters = phi_s
        self.parameters = phi_w
        print "Two-step CCA complete. Correlations between combined context and tokens (with rank %d): "%rank
        print cw_correlations

    def score(self, context_rep, phrase):
        bidi_con_rep = context_rep.dot(self.context_parameters)
        bidi_con_rep_norm = np.linalg.norm(bidi_con_rep)
        translations = self.inventory[phrase] #this pulls up all phrase pairs, incl. singletons
        scored_pairs = []
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            representation = self.get_representation(phrase_pair) #if representation is None, score is None
            score = None if representation is None else representation.dot(bidi_con_rep.transpose()) / (np.linalg.norm(representation)*bidi_con_rep_norm)
            #if score < 0:
            #    score = 0
            scored_pairs.append((phrase_pair, score))
        return scored_pairs

class GLM(BaseModel):
    def __init__(self, context, type_map, all_pp):
        super(GLM, self).__init__(type_map, all_pp, context)
        self.alphas = None

    def train(self, left_low_rank, right_low_rank, tokens, gamma, train_idxs=None):
        offset = np.ones((left_low_rank.shape[0], 1))        
        training_data = np.concatenate((left_low_rank, right_low_rank, offset), axis=1) #if each low-rank context is p-dim, then this is a 2p+1 dim vec
        self.parameters = compute_regression(training_data, tokens.get_token_matrix(train_idxs), gamma)
        print "Fitted general linear model: %d responses %d predictors, %d samples"%(self.parameters.shape[1], self.parameters.shape[0], left_low_rank.shape[0])

    def score(self, context_rep, phrase): 
        col_idxs, phrase_pairs = self.get_candidate_indices(phrase) #some of the col_idxs may be -1 
        aug_context_rep = np.append(context_rep, np.ones((1,)))
        if self.alphas is not None: #then element-wise multiply with alpha vec
            assert self.alphas.shape == aug_context_rep.shape
            aug_context_rep *= self.alphas
        scored_pps = []
        for real_idx, phrase_pair in enumerate(phrase_pairs): #col_idxs has -1 mixed in, so can't just multiply regularly
            if col_idxs[real_idx] >= 0: #phrase pair can be scored
                score = aug_context_rep.dot(self.parameters[:,col_idxs[real_idx]])
                if score < 0:                    
                    score = 0
                scored_pps.append((phrase_pair, score))
            else:
                scored_pps.append((phrase_pair, None))        
        return scored_pps

    def shrink_estimates(self, left_low_rank, right_low_rank, tokens, heldout_idxs): 
        labels = tokens.get_token_matrix(heldout_idxs)
        assert left_low_rank.shape[0] == right_low_rank.shape[0] == labels.shape[0]
        num_samples = left_low_rank.shape[0]
        offset = np.ones((left_low_rank.shape[0], 1))
        heldout_data = np.concatenate((left_low_rank, right_low_rank, offset), axis=1)
        start = time.clock()
        training_data = [[] for i in range(heldout_data.shape[1])] #list of p lists, where p is # of cols of heldout_data
        Y = []
        nonsingleton_count = 0
        for idx in xrange(num_samples):
            rows, cols = labels[idx,:].nonzero()
            assert len(cols) == 1
            phrase_id = cols[0] #if singleton, then this could be the OOV column
            phrase_pair = tokens.get_token_phrase(phrase_id) #if singleton, this could be <unk> ||| <unk>
            src_phrase = phrase_pair.split(' ||| ')[0]
            if src_phrase != "<unk>": #otherwise phrase pair is not in model, so can't use
                nonsingleton_count += 1
                translation_idxs, dummy = self.get_candidate_indices(src_phrase) #phrase_id is in translation_idxs --> get other candidates
                if len(translation_idxs) > 1: #want shrinkage estimator to disambiguate, so it's useless if there's only one translation for a phrase
                    ans_idx = translation_idxs.index(phrase_id)
                    one_hot = np.zeros((len(translation_idxs),))
                    one_hot[ans_idx] = 1.
                    Y.append(one_hot)
                    context_rep = heldout_data[idx,:]
                    for j in xrange(len(context_rep)): #features are indexed by j
                        beta_jk_vec = self.parameters[j,translation_idxs] #returns 1xh2 vec, where h2 is len(translation_idxs)
                        scaled_beta = context_rep[j]*beta_jk_vec #element-wise multiplication
                        training_data[j].append(scaled_beta) #append as training data
        Y_all = np.concatenate(Y)
        print "Training data size for shrinkage estimation: %d (from %d examples that aren't singletons)"%(Y_all.shape[0], nonsingleton_count)
        alphas = []
        for feat_vals in training_data: #each 'feat_vals' is a list of beta_jk * x_j column vectors for each j
            X = np.concatenate(feat_vals)
            assert X.shape == Y_all.shape
            alpha_j = (1./np.dot(X, X))*np.dot(X,Y_all) #alpha computation via normal equations (OLS)
            alphas.append(alpha_j)
        self.alphas = np.array(alphas)
        print "Shrinkage complete. Time taken: %.3f"%(time.clock()-start)
        print self.alphas

class MLP(BaseModel):
    def __init__(self, context, type_map, all_pp, gamma, rank): #there will be other params for the MLP
        super(MLP, self).__init__(type_map, all_pp, context)
        self.mlp = MLPClassifier(n_hidden=rank, verbose=1, lr=gamma) #set batch size
    
    def train(self, left_low_rank, right_low_rank, tokens, train_idxs=None):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        start = time.clock()
        self.mlp.fit(training_data, tokens.get_token_matrix(train_idxs))
        print "MLP learning complete; input layer size: %d; hidden layer size: %d; output layer size: %d; time taken: %.1f sec"%(training_data.shape[1], self.mlp.n_hidden, tokens.get_token_matrix().shape[1], time.clock()-start)
        
    def score(self, context_rep, phrase):
        predict_vec = self.mlp.predict(context_rep)        
        if predict_vec.shape[0] == 1: #single row vector
            predict_vec = np.reshape(predict_vec, (predict_vec.shape[1],))
        translations = self.inventory[phrase]
        scored_pairs = []
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            pp_id = self.get_tokenID(phrase_pair)
            if pp_id > 0:
                score = vec[pp_id]
                if score < 0:
                    score = 0
                scored_pairs.append((phrase_pair, score))
            else:
                scored_pairs.append((phrase_pair, None))
        return scored_pairs



        
