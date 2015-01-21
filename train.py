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
from sparsecontainer import *
from model import *

'''
extracts phrase pairs (tokens) and context, since the phrase pair list
is decorated with span information for the phrases so we can do this easily. 
'''
def extract_tokens(filehandle, sentence, tokens, left_con, right_con, extractor):
    sentence_items = sentence.split()
    for rule in filehandle:
        elements = rule.strip().split(' ||| ')
        src_phrase = elements[0]
        tgt_phrase = elements[1]
        phrase_pair = ' ||| '.join([src_phrase, tgt_phrase])
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
    
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'cf:g:k:l:m:o:ps:r:')
    phrase_pairs_loc = args[0]
    output_loc = args[1]
    con_length = 2
    rank1 = 100
    rank2 = 50
    gamma1 = 1e-5
    gamma2 = gamma1
    pos_depend = False
    filter_features = ""
    filter_sw = ""
    kappa = -1
    oov = 1 #cut-off for OOV handling of features
    method = "cca" #2-step cca is default
    concat = False
    for opt in opts:
        if opt[0] == '-l': #context length 
            con_length = int(opt[1])
        elif opt[0] == '-r': #ranks
            ranks = opt[1].split(',')
            if len(ranks) > 1: #rank decoupling
                rank1 = int(ranks[0])
                rank2 = int(ranks[1])
            else:
                rank1 = int(opt[1])
                rank2 = rank1
        elif opt[0] == '-g': #gammas (CCA/GLM regularization parameters)
            gammas = opt[1].split(',')
            if len(gammas) > 1:
                gamma1 = float(gammas[0])
                gamma2 = float(gammas[1])
            else:
                gamma1 = float(opt[1])
                gamma2 = gamma1
        elif opt[0] == '-p': #position-dependent feature extraction
            pos_depend = True
        elif opt[0] == '-f': #filter feature space; only consider words in this list
            filter_features = opt[1]
        elif opt[0] == '-s': #stop word filtering
            filter_sw = opt[1]
        elif opt[0] == '-k': #re-scaling
            kappa = float(opt[1])
        elif opt[0] == '-o': #OOV handling - set to singletons by default
            oov = int(opt[1])
        elif opt[0] == '-m': #method - one of 'cca', 'glm', or 'mlp'
            method = opt[1]
        elif opt[0] == '-c': #concatenation models (instead of additive)
            concat = True
    if phrase_pairs_loc == "" or output_loc == "":
        sys.stderr.write("Error! Need to define a location for per-sentence phrase pairs and an output directory to write parameters\n")
        sys.exit() 

    #set up data structures and extract context features and tokens from corpus
    start = time.clock()
    tokens = SparseContainer()
    left_con = SparseContext(oov)
    right_con = SparseContext(oov)
    extractor = ContextExtractor(con_length, pos_depend, filter_sw, filter_features)
    for count, line in enumerate(sys.stdin):
        source = line.strip().split(' ||| ')[0]
        phrase_pair_fh = gzip.open(phrase_pairs_loc + "/grammar.%d.gz"%count)
        extract_tokens(phrase_pair_fh, source, tokens, left_con, right_con, extractor)
        if count % 10000 == 0:
            print "Sentence Count: %d\t"%count,
            sys.stdout.flush()
        phrase_pair_fh.close()
    print
    print "Extracted features from parallel corpus. Time: %.1f sec"%(time.clock()-start)

    #convert context features to sparse matrix, rescale and compute parameters
    tokens.create_sparse_matrix() 
    left_con.create_sparse_matrix(pos_depend, con_length) #take pos depend and con_length as args to make pos-dep OOV types
    right_con.create_sparse_matrix(pos_depend, con_length) #can probably simplify and remove dependence on con_length (inferred from data)
    print "Number of tokens: %d; Number of types: %d"%(tokens.get_token_matrix().shape[0], tokens.get_token_matrix().shape[1])
    print "Left context dimensionality: %d; Right context dimensionality: %d"%(left_con.get_token_matrix().shape[1], right_con.get_token_matrix().shape[1])
    if kappa > 0: #rescaling requested; do we also rescale tokens? 
        left_con.rescale_features(kappa)
        right_con.rescale_features(kappa)
        print "Rescaled features with kappa=%.3f"%kappa
    context = Context(left_con, right_con, gamma1, rank1, concat, con_length)
    lr_mat_l, lr_mat_r = context.compute_lowrank_training_contexts(left_con.get_token_matrix(), right_con.get_token_matrix())
    model = None
    if method == "cca":
        model = CCA(context, tokens.get_type_map())
        model.train(lr_mat_l, lr_mat_r, tokens, gamma2, rank2)
    elif method == "glm":
        model = GLM(context, tokens.get_type_map())
        model.train(lr_mat_l, lr_mat_r, tokens, gamma2)
    elif method == "mlp":
        model = MLP(context, tokens.get_type_map(), rank2)
        model.train(lr_mat_l, lr_mat_r, tokens)
    else:
        sys.stderr.write("Model argument not recognized; please input one of 'cca', 'glm', or 'mlp'\n")
        sys.exit()

    #score model here on held-out
    start = time.clock()
    out_fh = open(output_loc, 'wb')
    cPickle.dump(model, out_fh)
    cPickle.dump(extractor, out_fh)
    out_fh.close()
    print "Wrote parameters to disk; Time taken: %.3f sec"%(time.clock()-start)

if __name__ == "__main__":
    main()
