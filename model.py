#!/usr/bin/python -tt

import sys, commands, string, os, time, gzip, cPickle, math
import numpy as np
import scipy.sparse as sp
import scipy.io as io
from scipy.special import expit
from mlp import MLPClassifier

'''
To Do: migrate MLP to VW setup
Also, clean up and figure out a better way to write out large .mat files for Matlab computation
'''


def compute_regression(X, Y, reg_strength):
    pwd = os.getcwd()
    out_locX1 = pwd + "/x1"
    out_locX2 = pwd + "/x2"
    N = X.shape[0] / 2 #file is large: split into 2
    io.savemat(out_locX1, {'X1': X[:N,:]})
    io.savemat(out_locX2, {'X2': X[N:,:]})
    out_locY = pwd + "/y"
    io.savemat(out_locY, {'Y': Y})
    path = os.path.abspath(os.path.dirname(sys.argv[0])) #matlab scripts in the same place as this script
    os.chdir(path+"/matlab")
    command = 'matlab -nodesktop -nosplash -nojvm -r "regression_wrapper ' + "%s %s %s %s"%(out_locX1, out_locX2, out_locY, reg_strength) + '"'
    os.system(command)
    os.chdir(pwd)
    mat_return = io.loadmat(out_locY)
    return mat_return['beta']

def compute_cca(X, Y, reg_strength, rank, approx, mean_center):
    p1 = X.shape[1]
    p2 = Y.shape[1]
    n = X.shape[0]
    #X = sp.csr_matrix(X) 
    cross_cov = None
    if approx != "full":
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
        #do we need to change below for dense matrices? 
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
    if approx == "full": #X can be large, especially in dense case - need a long-term solution to this!
        N = X.shape[0] / 8
        out_locX1 = pwd + "/x1"
        out_locX2 = pwd + "/x2"
        out_locX3 = pwd + "/x3"
        out_locX4 = pwd + "/x4"
        out_locX5 = pwd + "/x5"
        out_locX6 = pwd + "/x6"
        out_locX7 = pwd + "/x7"
        out_locX8 = pwd + "/x8"
        io.savemat(out_locX1, {'X1': X[:N,:]})
        io.savemat(out_locX2, {'X2': X[N:2*N,:]})
        io.savemat(out_locX3, {'X3': X[2*N:3*N,:]})
        io.savemat(out_locX4, {'X4': X[3*N:4*N,:]})
        io.savemat(out_locX5, {'X5': X[4*N:5*N,:]})
        io.savemat(out_locX6, {'X6': X[5*N:6*N,:]})
        io.savemat(out_locX7, {'X7': X[6*N:7*N,:]})
        io.savemat(out_locX8, {'X8': X[7*N:,:]})
        out_locY1 = pwd + "/y1"
        out_locY2 = pwd + "/y2"
        out_locY3 = pwd + "/y3"
        out_locY4 = pwd + "/y4"
        out_locY5 = pwd + "/y5"
        out_locY6 = pwd + "/y6"
        out_locY7 = pwd + "/y7"
        out_locY8 = pwd + "/y8"
        io.savemat(out_locY1, {'Y1': Y[:N,:]})
        io.savemat(out_locY2, {'Y2': Y[N:2*N,:]})
        io.savemat(out_locY3, {'Y3': Y[2*N:3*N,:]})
        io.savemat(out_locY4, {'Y4': Y[3*N:4*N,:]})
        io.savemat(out_locY5, {'Y5': Y[4*N:5*N,:]})
        io.savemat(out_locY6, {'Y6': Y[5*N:6*N,:]})
        io.savemat(out_locY7, {'Y7': Y[6*N:7*N,:]})
        io.savemat(out_locY8, {'Y8': Y[7*N:,:]})
        path = os.path.abspath(os.path.dirname(sys.argv[0])) 
        os.chdir(path+"/matlab") #matlab scripts in 'matlab' sub-directory
        os.system('matlab -nodesktop -nosplash -nojvm -r "cca_wrapper ' + "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"%(out_locX1, out_locX2, out_locX3, out_locX4, out_locX5, out_locX6, out_locX7, out_locX8, out_locY1, out_locY2, out_locY3, out_locY4, out_locY5, out_locY6, out_locY7, out_locY8, rank, reg_strength) + '"')  
    else:
        io.savemat(out_loc, {'avgOP': cross_cov})
        path = os.path.abspath(os.path.dirname(sys.argv[0])) 
        os.chdir(path+"/matlab")
        os.system('matlab -nodesktop -nosplash -nojvm -r "matlab_svd ' + out_loc + " %s"%rank + '"')
    os.chdir(pwd)
    out_locY1 = pwd + "/y1"
    mat_return = io.loadmat(out_locY1) if approx == "full" else io.loadmat(out_loc)
    os.system('rm *.mat') #clean up .mat files
    if approx == "diag": #if diag approx, multiply result by scaling
        return whiten_left.dot(mat_return['U']), whiten_right.dot(mat_return['V']), mat_return['S']
    else: #for full CCA, already scaled in cca_direct.m
        return mat_return['U'], mat_return['V'], mat_return['S']         

class Context:
    def __init__(self, left_context, left_type, right_context, right_type, gamma, rank, approx, mean_center, concat, con_length):
        phi_l, phi_r, lr_correlations = compute_cca(left_context, right_context, gamma, rank, approx, mean_center)
        self.left_proj_mat = phi_l
        self.right_proj_mat = phi_r
        self.left_map = left_type
        self.right_map = right_type
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
    def __init__(self, type_map, all_pp, context, is_vw):
        self.type_id_map = type_map
        self.parameters = None
        self.context = context
        self.inventory = self.__format_phrase_pairs(list(all_pp))
        self.is_vw = is_vw
        self.discretize_context = False
        self.discretize_phrasereps = False

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

    def set_context_means(self, pos_means, neg_means):
        self.context.pos_means = pos_means
        self.context.neg_means = neg_means
        self.discretize_context= True
        
    def set_phraserep_means(self, pos_means, neg_means):
        self.pos_means = pos_means
        self.neg_means = neg_means
        self.discretize_phrasereps = True

    def isvw(self):
        return self.is_vw

    def rep2str(self, rep, prefix):
        str_rep = []
        for idx,val in enumerate(rep):
            if prefix == "c" and self.discretize_context: 
                if val > 0:
                    new_val = 0 if val <= self.context.pos_means[idx] else 1
                else:
                    new_val = 0 if val >= self.neg_means[idx] else 1
                str_rep.append("%s_dim%d=%d"%(prefix,idx,new_val))
            elif prefix == "pp" and self.discretize_phrasereps:
                if val > 0:
                    new_val = 0 if val <= self.pos_means[idx] else 1
                else:
                    new_val = 0 if val >= self.neg_means[idx] else 1
                str_rep.append("%s_dim%d=%d"%(prefix,idx,new_val))
            else: #output raw value
                str_rep.append("%s_dim%d=%.3g"%(prefix,idx,val))
        return ' '.join(str_rep)

class CCA(BaseModel):
    def __init__(self, context, type_map, all_pp):
        super(CCA, self).__init__(type_map, all_pp, context, False)
        self.context_parameters = None

    def train(self, left_low_rank, right_low_rank, training_labels, gamma, rank, approx, mean_center):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        phi_s, phi_w, cw_correlations = compute_cca(training_data, training_labels, gamma, rank, approx, mean_center)
        self.context_parameters = phi_s
        self.parameters = phi_w
        print "Two-step CCA complete. Correlations between combined context and tokens (with rank %d): "%rank
        print cw_correlations

    def score(self, context_rep, phrase, print_reps):
        bidi_con_rep = context_rep.dot(self.context_parameters)
        bidi_con_rep_norm = np.linalg.norm(bidi_con_rep)
        translations = self.inventory[phrase] #this pulls up all phrase pairs, incl. singletons
        scored_pairs = []
        rep_str = ""
        if print_reps:
            rep_str += self.rep2str(bidi_con_rep, "c")
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            representation = self.get_representation(phrase_pair) #if representation is None, score is None
            rep_norm = np.linalg.norm(representation) if representation is not None else 0
            score = None if (representation is None or rep_norm == 0) else representation.dot(bidi_con_rep.transpose()) / (rep_norm*bidi_con_rep_norm)
            if score is not None: #add the phrase pair representation also
                rep_str_trans = rep_str +  " " + self.rep2str(representation, "pp") if print_reps else ""
                scored_pairs.append((phrase_pair, score, rep_str_trans))
            else:
                scored_pairs.append((phrase_pair, score, rep_str))
        return scored_pairs

class SVM(BaseModel):
    def __init__(self, context, type_map, all_pp):
        super(SVM, self).__init__(type_map, all_pp, context, False)

    def train(self, left_low_rank, right_low_rank, training_labels, gamma, out_dir):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        N = training_labels.shape[0]
        out_fh = open(out_dir + "/svm.training", 'wb')
        for row_idx in range(N):        
            label_row, label_col = training_labels[row_idx,:].nonzero()
            if len(label_row) == 1: #could be zero row
                phrase_id = label_col[0] + 1
                feature_str = ["%d:%.5g"%(idx+1,val) for idx,val in enumerate(training_data[row_idx,:])]
                print >> out_fh, '%d %s'%(phrase_id, ' '.join(feature_str))
        out_fh.close()
    
class MLR(BaseModel):
    def __init__(self, context, type_map, all_pp, ldf, high_dim):
        super(MLR, self).__init__(type_map, all_pp, context, True)
        self.ldf = {} if ldf else None
        self.model_loc = ""
        self.high_dim = high_dim

    def __label_features(self, features, src_id):
        left_row, right_row = np.split(features,2) #will this work with high_dim? 
        left_labeled = [("l%d"%(idx+1), val)  for idx,val in enumerate(left_row)] 
        right_labeled = [("r%d"%(idx+1), val)  for idx,val in enumerate(right_row)]
        feature_str = ' '.join(["%s_%d:%.5g"%(feat_label, src_id, feat_val) for feat_label,feat_val in left_labeled+right_labeled]) if src_id > -1 else ' '.join(["%s:%.5g"%(feat_label, feat_val) for feat_label,feat_val in left_labeled+right_labeled])
        return feature_str

    def __label_features_sparse(self, left, right):
        dummy, cols_left = left.nonzero()
        left_labeled = ["L%d"%col_idx for col_idx in cols_left]
        dummy, cols_right = right.nonzero()
        right_labeled = ["R%d"%col_idx for col_idx in cols_right]
        return ' '.join(left_labeled + right_labeled)
    
    def __decorate_labels(self, phrase_id, src_phrase, pp_counts, uniform_cost):
        col_idxs, phrase_pairs = self.get_candidate_indices(src_phrase) # over here, breaks when unk
        col_idxs = [idx for idx in col_idxs if idx != -1] #remove candidates that we can't score
        normalizer = np.sum(pp_counts[0,col_idxs]) - pp_counts[0,phrase_id] 
        if normalizer == 0 and len(col_idxs) > 1:
            normalizer = len(col_idxs)-1
        labels = []
        idx_in_col_idxs = False
        for idx in col_idxs:
            if idx == phrase_id:
                labels.append("%d:0.0"%idx)
                idx_in_col_idxs = True
            else:                
                #cost = 1.0 if uniform_cost else float(pp_counts[0,idx] + 1)
                #cost = 1./ len(col_idxs) if uniform_cost else float(pp_counts[0,idx]) / normalizer
                cost = 1.0 if uniform_cost else float((pp_counts[0,idx]+1)*len(col_idxs)) / normalizer #scale normalized counts by num translations
                labels.append("%d:%.1f"%(idx,cost))        
        assert idx_in_col_idxs == True
        return ' '.join(labels)

    def train(self, left_low_rank, right_low_rank, training_labels, gamma, id_type_map, out_dir, uniform_cost):
        marker = "ldf" if self.ldf is not None else "cost"
        if uniform_cost:
            marker += ".uniform"
        out_loc = out_dir + "/vw_train.%s.gz"%marker
        pp_counts = training_labels.sum(axis=0) #cost function for negative examples depends on this            
        N = training_labels.shape[0]             
        start = time.clock()
        if not os.path.isfile(out_loc): #if training data for VW has not been written out to file
            out_fh = gzip.open(out_loc, 'wb')
            training_data = np.hstack((left_low_rank, right_low_rank)) #should work with both dense and sparse matrices
            #training_data = np.concatenate((left_low_rank, right_low_rank), axis=1) #will this work if inputs are high dim??
            for row_idx in xrange(N): #may want to add a counter here
                label_row, label_col = training_labels[row_idx,:].nonzero()
                if len(label_col) == 1: #can be zero rows
                    phrase_pair = id_type_map[label_col[0]]
                    src_phrase = phrase_pair.split(' ||| ')[0]
                    src_id = -1
                    if self.ldf is not None:
                        if src_phrase not in self.ldf:
                            src_id = len(self.ldf)
                            self.ldf[src_phrase] = src_id
                        src_id = self.ldf[src_phrase]
                    feature_str = self.__label_features_sparse(left_low_rank[row_idx,:], right_low_rank[row_idx,:]) if self.high_dim else self.__label_features(training_data[row_idx,:], src_id)
                    label_str = self.__decorate_labels(label_col[0], src_phrase, pp_counts, uniform_cost) if src_phrase != "<unk>" else "%d:0.0"%label_col[0]
                    if self.ldf is not None: #write out in multi-line format
                        labels = label_str.split()
                        #out_fh.write("shared | %s\n"%feature_str)
                        for label in labels:
                            out_fh.write("%s | %s\n"%(label, feature_str))
                        out_fh.write("\n")
                    else:
                        out_fh.write("%s | %s\n"%(label_str, feature_str))
            out_fh.close()
            os.system('zcat %s | shuf > %s/temp; gzip %s/temp; mv %s/temp.gz %s'%(out_loc, out_dir, out_dir, out_dir, out_loc)) #shuffle data
            if self.ldf is not None: #write out dictionary that maps source phrases to IDs (for feature decoration in LDF)
                out_fh = open(out_dir+"/vw_ldf.srcphr.dict", 'wb')
                cPickle.dump(self.ldf, out_fh)
                out_fh.close()
                print "For label-dependent features: wrote out src phrase to ID dictionary"
            print "Assembled training data into VW format and wrote out to %s. Starting VW training..."%out_loc
        else:
            if self.ldf is not None: #load dictionary that maps source phrases to IDs (for feature decoration in LDF)
                in_fh = open(out_dir+"/vw_ldf.srcphr.dict", 'rb')
                self.ldf = cPickle.load(in_fh)
                in_fh.close()
                print "For label-dependent features: read in src phrase to ID dictionary"
            print "Already wrote training data to %s.  Starting VW training..."%out_loc

        self.model_loc = out_dir+"/mlr.%1g.model"%gamma
        #self.model_loc = out_dir+"/bfgs.model"
        if not os.path.isfile(self.model_loc): #if model does not exist, train it
            vw_command = ""
            if self.ldf is None:                
                vw_command = 'vw %s --compressed -c -f %s --csoaa %d --l2 %.1g --passes 2 --holdout_off'%(out_loc, self.model_loc, pp_counts.shape[1], gamma)
            else:
                num_features = left_low_rank.shape[1] + right_low_rank.shape[1]
                ring_size = int(math.ceil(math.log(num_features*len(self.ldf), 2)) + 1)
                vw_command = 'vw %s --compressed -c -f %s --csoaa_ldf mc --loss_function logistic --l2 %.1g -b %d --passes 2 --holdout_off'%(out_loc, self.model_loc, gamma, ring_size)
            assert vw_command != ""
            print "Running command: %s"%vw_command
            os.system(vw_command)
            print "VW training complete. Model located in %s"%self.model_loc
        else:
            print "Model %s already trained, no need to run VW training"%self.model_loc
        print "Multinomial Logistic Regression through VW complete. Time: %.1f sec"%(time.clock()-start)

    def __format_multiline(self, ml_pred, ml_scores):
        new_scores = []
        new_pred = []
        per_line_scores = []
        for idx,line in enumerate(ml_pred):
            if line.strip() != "":
                class_val = int(line.strip().split('.')[0])
                if class_val != 0: #predicted class
                    new_pred.append(class_val)
                label_str, score_str = ml_scores[idx].split(':')
                per_line_scores.append((int(label_str), float(score_str)))
            else: #assemble per_line_scores
                new_scores.append(per_line_scores)
                per_line_scores = []
        return new_pred, new_scores
        
    '''score_all takes a list of phrases and a matrix of context_reps and returns a list of lists: each list contains phrase pair-score tuples'''
    def score_all(self, context_reps, phrases, sent_num, print_reps):
        assert context_reps.shape[0] == len(phrases)
        dir_loc = os.path.dirname(self.model_loc) #write temporary files in the same directory as model
        file_id = str(sent_num)
        ex_filename = dir_loc + "/test.examples.%s"%file_id
        ex_predictions = dir_loc + "/test.predictions.%s"%file_id
        ex_scores = dir_loc + "/test.scores.%s"%file_id
        ex_fh = open(ex_filename, 'wb') #here: need to provide unique ID for filename
        for idx,phrase in enumerate(phrases): #write out all test examples to file
            context_rep = context_reps[idx,:] #context_rep could be sparse here
            col_idxs, phrase_pairs = self.get_candidate_indices(phrase)
            col_idxs = [idx for idx in col_idxs if idx != -1] #could be empty if all phrase pairs are less than threshold or have been pruned
            #need to handle empty col_idxs: occurs if all phrase pairs for source phrase are not in model
            label_str = ' '.join(["%d"%col_idx for col_idx in col_idxs])
            if self.ldf is not None: #label-dependent features
                labels = label_str.split()
                src_id = self.ldf[phrase] if phrase in self.ldf else 0 #will not be in ldf srcphrase dictionary if all examples of pp are in heldout
                feature_str = self.__label_features(context_rep, src_id)
                for label in labels:
                    print >> ex_fh, "%s | %s "%(label, feature_str)
                print >> ex_fh, ""
            else:
                feature_str = self.__label_features(context_rep, -1)
                print >> ex_fh, "%s | %s "%(label_str, feature_str)
        ex_fh.close()
        vw_command = 'vw --quiet -i %s -t -p %s -r %s < %s'%(self.model_loc, ex_predictions, ex_scores, ex_filename)
        os.system(vw_command)
        #if self.ldf is None: #for non LDF 
        #    os.system("sed -i '/^$/d' %s"%ex_scores) #just in case, NN outputs have empty lines

        pred_fh = open(ex_predictions, 'rb')
        predictions_raw = [line.strip() for line in pred_fh.readlines()]
        pred_fh.close()
        scores_fh = open(ex_scores, 'rb')
        scores_raw = [line.strip() for line in scores_fh.readlines()]
        scores_fh.close()
        predictions = []
        scores = []
        if self.ldf is not None:
            predictions, scores = self.__format_multiline(predictions_raw, scores_raw)
        else:
            predictions = [int(label.split('.')[0]) for label in predictions_raw]
            #below, why '-score'? because in VW, the score is the cost of each class, so higher means less probable
            for line in scores_raw:
                scores.append([(int(col_score_pair.split(':')[0]), -float(col_score_pair.split(':')[1])) for col_score_pair in line.split()])
                #scores.append([(int(col_score_pair.split(':')[0]), expit(-float(col_score_pair.split(':')[1]))) for col_score_pair in line.split()])
        assert len(predictions) == len(scores) == len(phrases)
        scored_pps_all = []
        for idx,scores_list in enumerate(scores): #convert VW outputs to list of phrase pair/score tuples
            phrase = phrases[idx]
            context_rep = context_reps[idx,:]
            col_idxs, phrase_pairs = self.get_candidate_indices(phrase)
            scored_pps = []            
            scored_dict = None
            if len(scores_list) > 0: #at least one score, so that means that at least one of the phrase pairs has been estimated
                prediction_score = max(scores_list, key = lambda x:x[1])[0] #what if it's flat? max outputs the first one
                prediction = predictions[idx]
                if prediction_score != prediction: #should add epsilon to the right answer because it's all flat right now
                    epsilon = 1e-5
                    new_scores_list = []
                    for phrase_id, score in scores_list:
                        if phrase_id == prediction: 
                            new_scores_list.append((phrase_id, score+epsilon))
                        else:
                            new_scores_list.append((phrase_id, score))
                    scores_list = new_scores_list
                scored_dict = dict(scores_list) 
            representation = self.rep2str(context_rep, "c") if print_reps else ""
            for real_idx, phrase_pair in enumerate(phrase_pairs): #convert to standard representation that interface expects
                if col_idxs[real_idx] >= 0: #phrase pair can be scored, and guaranteed that scored dict is not None
                    score = scored_dict[col_idxs[real_idx]]
                    scored_pps.append((phrase_pair, score, representation))
                else: #if all phrase pairs have None score, then scored dict is also None
                    scored_pps.append((phrase_pair, None, representation))
            scored_pps_all.append(scored_pps)
        os.system('rm %s; rm %s; rm %s'%(ex_filename, ex_predictions, ex_scores))
        return scored_pps_all

class GLM(BaseModel):
    def __init__(self, context, type_map, all_pp, lm_scores_add):
        super(GLM, self).__init__(type_map, all_pp, context, False)
        self.alphas = None
        self.lm_score = lm_scores_add

    def train(self, left_low_rank, right_low_rank, training_labels, gamma, lm_scores=None):        
        offset = np.ones((left_low_rank.shape[0], 1))
        if self.lm_score:
            lm_scores = lm_scores.reshape((lm_scores.shape[0], 1))
        training_data = np.concatenate((left_low_rank, right_low_rank, lm_scores, offset), axis=1) if self.lm_score else np.concatenate((left_low_rank, right_low_rank, offset), axis=1)
        self.parameters = compute_regression(training_data, training_labels, gamma)
        print "Fitted general linear model: %d responses %d predictors, %d samples"%(self.parameters.shape[1], self.parameters.shape[0], left_low_rank.shape[0])

    def score(self, context_rep, phrase, print_reps): 
        col_idxs, phrase_pairs = self.get_candidate_indices(phrase) #some of the col_idxs may be -1 
        aug_context_rep = np.concatenate((context_rep, np.zeros((1,)), np.ones((1,))), axis=1) if self.lm_score else np.append(context_rep, np.ones((1,)))
        if self.alphas is not None: #then element-wise multiply with alpha vec
            assert self.alphas.shape == aug_context_rep.shape
            aug_context_rep *= self.alphas
        scored_pps = []
        rep_str = ""
        if print_reps:
            rep_str += self.rep2str(context_rep, "c")
        for real_idx, phrase_pair in enumerate(phrase_pairs): #col_idxs has -1 mixed in, so can't just multiply regularly
            if col_idxs[real_idx] >= 0: #phrase pair can be scored
                parameter = self.parameters[:,col_idxs[real_idx]]
                score = aug_context_rep.dot(parameter)
                rep_str_trans = rep_str + " " + self.rep2str(parameter, "pp") if print_reps else ""
                scored_pps.append((phrase_pair, score, rep_str_trans))
            else:
                scored_pps.append((phrase_pair, None, rep_str))        
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
        print "Shrinkage complete. Time: %.1f sec"%(time.clock()-start)
        print self.alphas

class MLP(BaseModel):
    def __init__(self, context, type_map, all_pp, gamma, rank): #there will be other params for the MLP
        super(MLP, self).__init__(type_map, all_pp, context, False)
        self.mlp = MLPClassifier(n_hidden=rank, verbose=1, lr=gamma) #set batch size
    
    def train(self, left_low_rank, right_low_rank, training_labels):
        training_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
        start = time.clock()
        self.mlp.fit(training_data, training_labels)
        print "MLP learning complete; input layer size: %d; hidden layer size: %d; output layer size: %d; time: %.1f sec"%(training_data.shape[1], self.mlp.n_hidden, tokens.get_token_matrix().shape[1], time.clock()-start)
        
    def score(self, context_rep, phrase, print_reps):
        predict_vec = self.mlp.predict(context_rep)        
        if predict_vec.shape[0] == 1: #single row vector
            predict_vec = np.reshape(predict_vec, (predict_vec.shape[1],))
        translations = self.inventory[phrase]
        scored_pairs = []
        rep_str = ""
        if print_reps:
            rep_str += self.rep2str(context_rep, "c")
        for translation in translations:
            phrase_pair = ' ||| '.join([phrase, translation])
            pp_id = self.get_tokenID(phrase_pair)
            if pp_id > 0:
                score = vec[pp_id]
                scored_pairs.append((phrase_pair, score))
            else:
                scored_pairs.append((phrase_pair, None))
        return scored_pairs



        
