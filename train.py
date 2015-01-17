#!/usr/bin/python -tt

'''
File: training.py
Author: Avneesh Saluja (avneesh@cs.cmu.edu)
Date: December 23, 2014
Description: given a sentence-aligned parallel corpus and a list of phrase pairs associated
with each sentence pair, this script first assembles word and context co-occurrence matrices
on a token basis, which are extremely sparse matrices.  It then computes a CCA based on these
co-occurrence matrices, resulting in two projection matrices, one for context, and one for
phrase pairs, which brings them into the same space.  These projection matrices are written
out for subsequent decoding/evaluation. 
arg0: config file - defines output location of parameters (same config given to decoder)
arg1: directory location of sentence-specific phrase pairs (in .gz format; output of extract 
STDIN: training corpus (can be parallel or just source)
Flags:
-l arg: change context length (default is 2 i.e., 2 words on each side)
-r arg: change rank (default is 50); to decouple rank, provide two ranks separated by a comma e.g., '100,50'
-k arg: enable re-scaling of features based on diagonal approximation to covariance, with an additional scaling factor (default: disabled)
-p: make context features position-dependent
-s arg: filter stop words, provided in arg (list of words to remove)
-f arg: filter to words provided in arg; if they contain stop words and stop words is enabled, then they will be removed
-c arg: write out counts in cPickle format to location specified by arg
-o arg: compute OOV parameter by removing freatures with count <= arg and replacing them with <unk> token
'''

import sys, commands, string, gzip, getopt, os, cPickle, time
import sklearn.linear_model as lm
from eigentype import *
import scipy.io as io
import numpy as np

'''
extracts phrase pairs (tokens) and context, since the phrase pair list
is decorated with span information for the phrases so we can do this easily. 
'''
def extract_tokens(filehandle, sentence, tokens, left_con, right_con, extractor, count_dict):
    sentence_items = sentence.split()
    for rule in filehandle:
        elements = rule.strip().split(' ||| ')
        src_phrase = elements[0]
        tgt_phrase = elements[1]
        phrase_pair = ' ||| '.join([src_phrase, tgt_phrase])
        if count_dict is not None:
            tgt_counts = count_dict[src_phrase] if src_phrase in count_dict else {}
            tgt_counts[tgt_phrase] = tgt_counts[tgt_phrase] + 1 if tgt_phrase in tgt_counts else 1
            count_dict[src_phrase] = tgt_counts
        span = rule.strip().split(' ||| ')[2]
        left_idx = int(span.split('-')[0])
        right_idx = int(span.split('-')[1])
        left_con_words, right_con_words = extractor.extract_context(sentence_items, left_idx, right_idx)
        if len(left_con_words) > 0 and len(right_con_words) > 0:
            tokens.add_token([phrase_pair])
            left_con.add_token(left_con_words)
            right_con.add_token(right_con_words)
        else:
            sys.stderr.write("WARNING! Empty left and/or right context due to stop-word filtering. You may want to either a) reduce the number of stop words or disable stop word filtering altogether or b) enlarge the context window size with the '-l' flag.\n")
            sys.stderr.write("The phrase pair %s was skipped as a result\n"%phrase_pair)

'''
for CCA computations: X - left matrix, Y - right matrix, param - rank
for regression: X - desin matrix, Y - response matrix, param - regularization strength
'''
def matlab_interface(X, Y, reg_strength, option, rank = 50):
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

'''
preprocessing step for both 2-step CCA and low-dimensional regression approaches
'''
def compute_lowrank_context(left_con, right_con, gamma, rank):
    phi_l, phi_r, lr_correlations = matlab_interface(left_con.token_mat, right_con.token_mat, gamma, "CCA", rank) 
    left_con.set_projection_mat(phi_l)
    right_con.set_projection_mat(phi_r)
    print "Correlations between left and right context (with rank %d): "%rank
    print lr_correlations
    left_low_rank = left_con.get_lowrank_token_mat()
    right_low_rank = right_con.get_lowrank_token_mat()    
    return left_low_rank, right_low_rank

'''
maybe we should play around with the regularization parameter in CCA? set this as an option
does 2-step CCA computation
'''
def compute_cca(left_con, right_con, bidi_lowrank_con, tokens, rank1, rank2): 
    gamma = 1e-8
    left_low_rank, right_low_rank = compute_lowrank_context(left_con, right_con, gamma, rank1)
    bidi_lowrank_con.set_token_mat(np.concatenate((left_low_rank, right_low_rank), axis=1))                                   
    phi_s, phi_w, cw_correlations = matlab_interface(bidi_lowrank_con.get_token_mat(), tokens.get_token_mat(), gamma, "CCA", rank2) 
    bidi_lowrank_con.set_projection_mat(phi_s)
    tokens.set_projection_mat(phi_w)
    print "Correlations between combined context and tokens (with rank %d): "%rank2
    print cw_correlations

'''
first computes low-dimensional context representation using CCA then uses the
resulting repreesentations as predictors and regresses class indicator matrix
against them. 
'''
def compute_regression(left_con, right_con, tokens, rank):
    gamma = 1e-8
    left_low_rank, right_low_rank = compute_lowrank_context(left_con, right_con, gamma, rank)
    offset = np.ones((left_low_rank.shape[0], 1))
    X_lowrank = np.concatenate((left_low_rank, right_low_rank, offset), axis=1) #if each low-rank context is p-dimensional, this is a 2p+1 dimensional vector    
    Y = tokens.get_token_mat()
    weights = matlab_interface(X_lowrank, Y, 1, "regression")
    print "Fitted ridge regression: %d responses, %d predictors"%(Y.shape[1], X_lowrank.shape[1])    
    #here: compute R^2 and output
    tokens.set_projection_mat(weights)    
    
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'c:f:k:l:o:ps:r:R')
    config_file = args[0]
    phrase_pairs_loc = args[1]
    con_length = 2
    rank1 = 50
    rank2 = 50
    pos_depend = False
    filter_features = ""
    filter_sw = ""
    kappa = -1
    oov = 0 #cut-off for OOV handling
    count_dict = None
    count_out_loc = ""
    regression = False #2-step CCA and k-NN by default
    for opt in opts:
        if opt[0] == '-l':
            con_length = int(opt[1])
        elif opt[0] == '-r':
            ranks = opt[1].split(',')
            if len(ranks) > 1: #rank decoupling
                rank1 = int(ranks[0])
                rank2 = int(ranks[1])
            else:
                rank1 = int(opt[1])
                rank2 = rank1
        elif opt[0] == '-p': #position-dependent feature extraction
            pos_depend = True
        elif opt[0] == '-f': #filter feature space; only consider words in this list
            filter_features = opt[1]
        elif opt[0] == '-s': #stop word filtering
            filter_sw = opt[1]
        elif opt[0] == '-k': #scale
            kappa = float(opt[1])
        elif opt[0] == '-c': #write out count dict
            count_dict = {}
            count_out_loc = opt[1]
        elif opt[0] == '-o': #OOV handling
            oov = int(opt[1])
        elif opt[0] == '-R': #regression
            regression = True
    if config_file == "" or phrase_pairs_loc == "": #error - need to define these arguments
        sys.stderr.write("Error! Need to define a config file and a location for per-sentence phrase pairs\n")
        sys.exit()
    tokens = eigentype()
    left_con = eigentype(oov)
    right_con = eigentype(oov)
    bidi_lowrank_con = eigentype()
    extractor = context_extractor(con_length, pos_depend, filter_sw, filter_features)
    for count, line in enumerate(sys.stdin):
        source = line.strip().split(' ||| ')[0]
        phrase_pair_fh = gzip.open(phrase_pairs_loc + "/grammar.%d.gz"%count)
        extract_tokens(phrase_pair_fh, source, tokens, left_con, right_con, extractor, count_dict)
        if count % 10000 == 0:
            print "Sentence Count: %d\t"%count,
            sys.stdout.flush()
        phrase_pair_fh.close()
    print
    print "Extracted features from parallel corpus"
    if count_dict is not None:
        cPickle.dump(count_dict, open(count_out_loc, "wb"))
        print "Wrote counts to file %s"%count_out_loc
    tokens.create_sparse_matrix() 
    left_con.create_sparse_matrix(pos_depend, con_length) #take pos depend as argument to create pos specific OOV
    right_con.create_sparse_matrix(pos_depend, con_length) #can probably simplify and remove dependence on con_length (inferred from data)
    print "Number of tokens: %d; Number of types: %d"%(tokens.token_mat.shape[0], tokens.token_mat.shape[1])
    print "Left context dimensionality: %d; Right context dimensionality: %d"%(left_con.token_mat.shape[1], right_con.token_mat.shape[1])
    if kappa > 0: #rescaling requested; do we also rescale tokens? 
        left_con.rescale_features(kappa)
        right_con.rescale_features(kappa)
        print "Rescaled features with kappa=%.3f"%kappa
    if regression:
        compute_regression(left_con, right_con, tokens, rank1)
        print "Low-dimensional regression complete"
    else:
        compute_cca(left_con, right_con, bidi_lowrank_con, tokens, rank1, rank2)
        print "CCA computation complete"
    start = time.clock()
    conf = config(config_file)
    left_con.write_to_file(conf.get_fileloc("left_con"))
    right_con.write_to_file(conf.get_fileloc("right_con"))
    if not regression:
        bidi_lowrank_con.write_to_file(conf.get_fileloc("lowrank_con"))
    tokens.write_to_file(conf.get_fileloc("tokens"))
    cPickle.dump(extractor, open(conf.get_fileloc("context_extractor"), "wb"))
    print "Wrote parameters to disk; Time taken: %.3f sec"%(time.clock()-start)

if __name__ == "__main__":
    main()
