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
arg0: directory location of sentence-specific phrase pair lists
arg1: output location of parameters
STDIN: training corpus (can be parallel or just source)
Flags:
-l: change context length (default is 2 i.e., 2 words on each side)
To Do:
- add flag to restrict context/features to N most frequent words. 
- add flag for stop word filtering (remove N top words)
- add flag for feature scaling before CCA
'''

import sys, commands, string, gzip, getopt, os, cPickle
from eigentype import *
import scipy.io as io
import numpy as np

def extract_tokens(filehandle, sentence, tokens, left_con, right_con, con_length, pos_depend):
    sentence_items = sentence.split()
    extractor = context_extractor(con_length, pos_depend)
    for rule in filehandle:
        phrase_pair = ' ||| '.join(rule.strip().split(' ||| ')[:2])
        span = rule.strip().split(' ||| ')[2]
        tokens.add_token([phrase_pair])
        left_idx = int(span.split('-')[0])
        right_idx = int(span.split('-')[1])
        left_con_words, right_con_words = extractor.extract_context(sentence_items, left_idx, right_idx)
        left_con.add_token(left_con_words)
        right_con.add_token(right_con_words)

def matlab_interface(left_mat, right_mat, rank):
    pwd = os.getcwd()
    out_loc = pwd + "/matlab_temp"
    io.savemat(out_loc, {'left': left_mat, 'right': right_mat})
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    os.chdir(path)
    os.system('matlab -nodesktop -nosplash -nojvm -r "matlab_wrapper ' + out_loc + " %s"%rank + '"')
    os.chdir(pwd)
    mat_return = io.loadmat(out_loc)
    return mat_return['A'].newbyteorder('='), mat_return['B'].newbyteorder('='), mat_return['r'].newbyteorder('=')

def compute_cca(left_con, right_con, bidi_lowrank_con, tokens, rank): 
    phi_l, phi_r, lr_correlations = matlab_interface(left_con.token_mat, right_con.token_mat, rank)
    left_con.projection_mat = phi_l
    right_con.projection_mat = phi_r
    print "Correlations between left and right context: "
    print lr_correlations
    left_low_rank = left_con.project(left_con.token_mat)
    right_low_rank = right_con.project(right_con.token_mat)
    bidi_lowrank_con.token_mat = np.concatenate((left_low_rank, right_low_rank), axis=1)
    phi_s, phi_w, cw_correlations = matlab_interface(bidi_lowrank_con.token_mat, tokens.token_mat, rank)
    bidi_lowrank_con.projection_mat = phi_s
    tokens.projection_mat = phi_w
    print "Correlations between combined context and tokens: "
    print cw_correlations
    
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'k:l:p')
    phrase_pairs_loc = args[0]
    output_loc = args[1]
    con_length = 2
    rank = 50
    pos_depend = False
    for opt in opts:
        if opt[0] == '-l':
            con_length = int(opt[1])
        elif opt[0] == '-k':
            rank = int(opt[1])
        elif opt[0] == '-p': #position-dependent feature extraction
            pos_depend = True

    tokens = eigentype()
    left_con = eigentype()
    right_con = eigentype()
    bidi_lowrank_con = eigentype()    
    for count, line in enumerate(sys.stdin):
        source = line.strip().split(' ||| ')[0]
        phrase_pair_fh = gzip.open(phrase_pairs_loc + "/grammar.%d.gz"%count)
        extract_tokens(phrase_pair_fh, source, tokens, left_con, right_con, con_length, pos_depend)
        phrase_pair_fh.close()
    tokens.create_sparse_matrix()
    left_con.create_sparse_matrix()
    right_con.create_sparse_matrix()
    print "Number of tokens: %d; Number of types: %d"%(tokens.token_mat.shape[0], tokens.token_mat.shape[1])
    print "Left context dimensionality: %d; Right context dimensionality: %d"%(left_con.token_mat.shape[1], right_con.token_mat.shape[1])
    compute_cca(left_con, right_con, bidi_lowrank_con, tokens, rank)
    paramDict = {}
    paramDict["left_con"] = left_con
    paramDict["right_con"] = right_con
    paramDict["lowrank_con"] = bidi_lowrank_con
    paramDict["tokens"] = tokens
    paramDict["con_length"] = con_length
    paramDict["pos_depend"] = pos_depend
    cPickle.dump(paramDict, open(output_loc, "wb"))

if __name__ == "__main__":
    main()
