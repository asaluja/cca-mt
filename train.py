#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

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
Flags: see README.md for more details
'''

import sys, commands, string, gzip, getopt, os, cPickle, time, random, gc
from scipy.special import expit
from sparsecontainer import *
from model import *
from source_channel import *

'''
extracts phrase pairs (tokens) and context, since the phrase pair list
is decorated with span information for the phrases so we can do this easily. 
excluded pairs is empty set if we are not filtering anything
'''
def extract_tokens(filehandle, sentence, tokens, left_con, right_con, extractor, all_phrase_pairs, excluded_pairs, use_wordvecs):
    sentence_items = sentence.split()
    for rule in filehandle:
        elements = rule.strip().split(' ||| ')
        src_phrase = elements[0]
        tgt_phrase = elements[1]
        phrase_pair = ' ||| '.join([src_phrase, tgt_phrase])
        if phrase_pair not in excluded_pairs: #only add to list of phrase pairs if it is a "valid" PP (e.g., in top N as per P(e|f))
            all_phrase_pairs.add(phrase_pair)
        span = rule.strip().split(' ||| ')[2]
        left_idx = int(span.split('-')[0])
        right_idx = int(span.split('-')[1])
        left_con_words, right_con_words = extractor.extract_context(sentence_items, left_idx, right_idx)
        if use_wordvecs: #left_con_words and right_con_words are two real-valued vectors
            tokens.add_token(phrase_pair, excluded_pairs)
            left_con.add_token_vec(left_con_words)
            right_con.add_token_vec(right_con_words)
        else: #left_con_words and right_con_words are two lists containing context words
            if len(left_con_words) > 0 and len(right_con_words) > 0:
                tokens.add_token(phrase_pair, excluded_pairs)
                left_con.add_token(left_con_words)
                right_con.add_token(right_con_words)
            else:
                sys.stderr.write("WARNING! Empty left and/or right context due to stop-word filtering. You may want to either a) reduce the number of stop words or disable stop word filtering altogether or b) enlarge the context window size with the '-l' flag.\n")
                sys.stderr.write("The phrase pair %s was skipped as a result\n"%phrase_pair)

def score_heldout(heldout_data, tokens, heldout_idxs, model, sc_model):
#def score_heldout(heldout_data, tokens, heldout_idxs, model, sc_model, lm_scores, counts, reverse_counts):
    phrases = []
    ground_truth = []
    example_idxs = []
    print "Heldout: beginning evaluation"
    start = time.clock()    
    for idx, test_idx in enumerate(heldout_idxs): #assemble phrases
        rows, cols = tokens.get_token_matrix()[test_idx,:].nonzero()
        if len(cols) == 1: #if it is not a zero row, otherwise it has been pruned
            phrase_pair = tokens.get_token_phrase(cols[0])            
            src_phrase = phrase_pair.split(' ||| ')[0]
            if src_phrase != "<unk>": #if pruned away, then cols[0] will equate to <unk> col
                phrases.append(src_phrase)
                ground_truth.append(phrase_pair)
                example_idxs.append(idx)
    print "Heldout: assembled phrases that can be scored"
    if sc_model is not None:
        print "Evaluating Source Channel model"
        scored_pps_lm, scored_pps_fwd, scored_pps_lm_fwd, scored_pps_rev, scored_pps_lm_rev = [], [], [], [], []
        for idx,src_phrase in enumerate(phrases): 
            lm, fwd, lm_fwd, rev, lm_rev = sc_model.score_all(example_idxs[idx], src_phrase)
            scored_pps_lm.append(lm)
            scored_pps_fwd.append(fwd)
            scored_pps_lm_fwd.append(lm_fwd)
            scored_pps_rev.append(rev)
            scored_pps_lm_rev.append(lm_rev)
        print "Heldout: scored phrases"
        print "LM MRR: "
        compute_mrr(scored_pps_lm, ground_truth, tokens, heldout_idxs)
        print "P(e|f) MRR: "
        compute_mrr(scored_pps_fwd, ground_truth, tokens, heldout_idxs)
        print "LM + P(e|f) MRR: "
        compute_mrr(scored_pps_lm_fwd, ground_truth, tokens, heldout_idxs)
        print "P(f|e) MRR: "
        compute_mrr(scored_pps_rev, ground_truth, tokens, heldout_idxs)
        print "LM + P(f|e) MRR: "
        compute_mrr(scored_pps_lm_rev, ground_truth, tokens, heldout_idxs)
    else:        
        scored_pps_all = [] #all src phrases
        if model.isvw(): #then score all phrases together
            #lm_scores_new = [lm_dict for idx,lm_dict in enumerate(lm_scores) if idx in example_idxs]
            #scored_pps_all = model.score_all(heldout_data[example_idxs,:], phrases, 0, False, lm_scores_new, counts, reverse_counts)
            scored_pps_all = model.score_all(heldout_data[example_idxs,:], phrases, 0, False)
        else:
            for idx,src_phrase in enumerate(phrases): #score each phrase individually            
                scored_pps = model.score(heldout_data[example_idxs[idx],:], src_phrase, False) 
                scored_pps_all.append(scored_pps)
        print "Heldout: scored phrases"
        compute_mrr(scored_pps_all, ground_truth, tokens, heldout_idxs)
    print "Heldout: computed MRR. Total time: %.1f sec"%(time.clock()-start)

def compute_mrr(scored_pps_all, ground_truth, tokens, heldout_idxs):
    mrr = 0
    mrr_hard_examples = 0
    multi_translation_count = 0
    hard_examples_count = 0
    avg_length = 0
    sq_loss = 0
    for idx,scored_pps in enumerate(scored_pps_all): #compute MRR for scored PPs
        if len(scored_pps) > 1: #only record MRR for src phrases with more than one translation
            avg_length += len(scored_pps)
            sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True) #sort by score to get rank
            phrase_pair = ground_truth[idx]
            best_score = [pp_score[1] for pp_score in sorted_pps if pp_score[0] == phrase_pair][0]
            if best_score is not None:
                scored_rank = [rank+1 for rank, pp_score in enumerate(sorted_pps) if pp_score[0] == phrase_pair][0] #because we filtered for singleton PPs, guaranteed that scored_rank is not None
                mrr += 1./scored_rank
                multi_translation_count += 1
                sq_loss += (1-best_score)**2 
                if scored_rank > 1: #separate tracking if correct translation was not ranked 1
                    mrr_hard_examples += 1./scored_rank
                    hard_examples_count += 1
            else: #this will happen if all instances of an unpruned phrase happen to be in the heldout set
                print "Phrase pair '%s' representation is None"%phrase_pair
                col_id = tokens.type_id_map[phrase_pair]
                rows, cols = tokens.get_token_matrix()[:,col_id].nonzero()
                assert len(rows) > 0                
                for row_idx in rows:
                    if row_idx not in heldout_idxs: 
                        print "Error! example idx for phrase pair is in training, not heldout!"
                        sys.exit()
                else:
                    print "All instances of this phrase pair are in heldout"
    mrr /= multi_translation_count
    sq_loss /= multi_translation_count
    mrr_hard_examples /= hard_examples_count
    avg_length = float(avg_length) / multi_translation_count
    print "Out of %d examples (source phrases), %d are not pruned (singleton/filtered), %d have > 1 translation (restricting scoring to these phrases)"%(len(heldout_idxs), len(scored_pps_all), multi_translation_count)
    print "Mean Reciprocal Rank: %.3f; Average number of translations per phrase: %.2f"%(mrr, avg_length)
    print "Mean Reciprocal Rank for phrase pairs not ranked 1: %.3f; Number of such examples: %d"%(mrr_hard_examples, hard_examples_count)
    print "Squared Error: %.3f"%sq_loss

def extract_excluded_PPs(counts_dict, cutoff):
    excluded_PPs = set()
    for src_rule in counts_dict:
        if len(counts_dict[src_rule]) > cutoff: #if number of translation options > cutoff, then prune
            sorted_translations = sorted(counts_dict[src_rule], key=counts_dict[src_rule].get, reverse=True)
            rules_to_filter = sorted_translations[cutoff:]
            original_count = len(counts_dict[src_rule])
            for rule in rules_to_filter:
                filtered_rule = ' ||| '.join([src_rule, rule]) #form phrase pair
                excluded_PPs.add(filtered_rule)
    return excluded_PPs

'''
function that populates the list 'lm_scores_all' passed in as an argument
lm_scores_all is indexed by the training example ID (row ID), and each element is
a dictionary of (target_phrase, LM score) pairs.  The data is used downstream
when we output features for our matching model. 
'''
def read_lm_scores(lm_phrases_loc, lm_scores_loc, cutoff, lm_scores_all):
    lm_ph_fh = open(lm_phrases_loc, 'rb')
    lm_phrases = lm_ph_fh.read().splitlines()
    lm_ph_fh.close()
    lm_scores_fh = open(lm_scores_loc, 'rb')
    lm_scores = lm_scores_fh.read().splitlines()
    lm_scores_fh.close()
    lm_scores_per_phrase = {}
    for idx,line in enumerate(lm_scores):
        val = float(line.strip().split()[1])
        if val < 0: #valid
            src_phrase, tgt_phrase, context = lm_phrases[idx].strip().split(' ||| ')
            num_words = len(context.split())
            lm_scores_per_phrase[tgt_phrase] = -val/num_words
        else: #finished reading in scores for source phrase
            assert lm_phrases[idx].strip() == ""
            #lm_scores_all.append(list(lm_scores_per_phrase))
            lm_scores_all.append(dict(lm_scores_per_phrase))
            lm_scores_per_phrase = {}

def convert_counts_dict(counts_dict):
    reverse = {}
    for src_phrase in counts_dict:
        for tgt_phrase in counts_dict[src_phrase]:
            source_dict = reverse[tgt_phrase] if tgt_phrase in reverse else {}
            source_dict[src_phrase] = counts_dict[src_phrase][tgt_phrase]
            reverse[tgt_phrase] = source_dict
    return reverse
    
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'a:A:bcC:dDf:F:g:h:Hl:Lm:Mo:OpP:s:S:r:t:uv:V:w:')
    phrase_pairs_loc = args[0]
    output_loc = args[1]
    con_length = 2
    rank1 = 100
    rank2 = 50
    gamma1 = 1
    gamma2 = gamma1
    pos_depend = False
    filter_features = ""
    filter_sw = ""
    filter_grammar = ""
    filter_cutoff = 0
    vector_loc = ""
    context_loc = ""
    tokens_loc = ""
    oov = 1 #cut-off for OOV handling of features; singleton words used to estimate OOV context parameter
    prune_tokens = 0
    estimate_oov_param = False
    method = "cca" #2-step cca is default
    whitening = "full" #regular CCA
    mean_center = False
    concat = False
    ldf = False
    uniform_cost = False
    heldout_frac = 0
    shrinkage_frac = 0
    high_dim = False
    lm_score_loc = ""
    matching_model = 0    
    relative_estimates = False
    use_lm_filter = False
    context_vec_loc = ""
    pp_vec_loc = ""
    source_channel = False
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
        elif opt[0] == '-F':
            filter_options = opt[1].split(',')
            if len(filter_options) != 2:
                sys.stderr.write("Error! If translation model filtering by P(e|f) requested, need to provide counts file and filter count threshold separated by comma ','\n")
                sys.exit()
            filter_grammar, filter_cutoff = filter_options[0], int(filter_options[1])
        elif opt[0] == '-s': #stop word filtering
            filter_sw = opt[1]
        elif opt[0] == '-o': #OOV handling - set to singletons by default
            oov = int(opt[1])
        elif opt[0] == '-P':
            prune_tokens = int(opt[1])
        elif opt[0] == '-O':
            estimate_oov_param = True
        elif opt[0] == '-m': #method - one of 'cca', 'glm', 'mlr', 'svm', or 'mlp'
            method = opt[1]
        elif opt[0] == '-c': #concatenation models (instead of additive)
            concat = True
        elif opt[0] == '-h': #held-out fraction
            heldout_frac = float(opt[1])
        elif opt[0] == '-S': #held-out fraction for shrinkage purposes
            shrinkage_frac = float(opt[1])
        elif opt[0] == '-v': #use pre-computed vector representations
            vector_loc = opt[1]
        elif opt[0] == '-C': #context file 
            context_loc = opt[1]
        elif opt[0] == '-t': #token file (for large datasets)
            tokens_loc = opt[1]
        elif opt[0] == '-w': 
            whitening = opt[1]
        elif opt[0] == '-M': #mean-center
            mean_center = True
        elif opt[0] == '-L': 
            ldf = True
        elif opt[0] == '-u': #uniform cost
            uniform_cost = True
        elif opt[0] == '-H': #high-dimensional
            high_dim = True
        elif opt[0] == '-a': 
            lm_options = opt[1].split(',')
            if len(lm_options) != 2:
                sys.stderr.write("Error! If LM score addition option (-a) selected, then need to provide the list of phrase pairs and context in one file, separated by their LM scores, separated by comma ','\n")
                sys.exit()
            lm_phrases_loc, lm_score_loc = lm_options[0], lm_options[1]
        elif opt[0] == '-A':
            matching_model = int(opt[1])
        elif opt[0] == '-b': #add relative freq estimates to matching model
            relative_estimates = True
        elif opt[0] == '-V': #read in word and phrase vectors for skip-gram model
            vector_options = opt[1].split(',')
            if len(vector_options) != 2:
                sys.stderr.write("Error! If word and phrase vectors from skip-gram selected (-V arg1,arg2) then need to provide the word vectors and the phrase vectors, with the filenames separated by a comma ','\n")
                sys.exit()
            context_vec_loc, pp_vec_loc = vector_options[0], vector_options[1]
        elif opt[0] == '-d': #source-channel model
            source_channel = True
            relative_estimates = True
        elif opt[0] == '-D': #use LM filter when learning matching model
            use_lm_filter = True
    if phrase_pairs_loc == "" or output_loc == "": #these inputs are required
        sys.stderr.write("Error! Need to define a location for per-sentence phrase pairs and an output directory to write parameters\n")
        sys.exit() 
    if filter_features != "" and vector_loc != "": 
        sys.stderr.write("Error! Restricting context features to a particular list ('-f X') is not supported with word vector representations ('-v X') yet\n")
        sys.exit()
    if concat and vector_loc != "":
        sys.stderr.write("Error! Concatenative models for context ('-c') and word vector representations ('-v X') don't make sense together. If you want concatenative word vector representations, set '-p' and '-v X' to get the effect (otherwise, the vectors get added\n")
        sys.exit()
    if heldout_frac > 0 and shrinkage_frac > 0:
        sys.stderr.write("Error! Cannot do shrinkage and held-out estimation at the same time; held-out set is used for James-Stein shrinkage, and mean squared error will be output on that set\n")
        sys.exit()
    if shrinkage_frac > 0 and method != "glm":
        sys.stderr.write("Error! James-Stein shrinkage only works on GLM (general linear model)\n")
        sys.exit()
    if not (whitening == "identity" or whitening == "full" or whitening == "diag" or whitening == "ppmi"):
        sys.stderr.write("Error! Whitening values can only be 'identity', 'full' (inverse square root), 'diag' (approximation to inverse square root), or 'ppmi' (PPMI scaling)\n")
        sys.exit()
    if not (method == "cca" or method == "glm" or method == "mlr" or method == "mlp" or method == "svm" or method == "sg"):
        sys.stderr.write("Error! Currently supported supervised methods are 'cca', 'glm', 'mlr', 'svm', 'mlp', or 'sg'\n")
        sys.exit()
    if prune_tokens == 0 and estimate_oov_param is True: 
        sys.stderr.write("Error! OOV parameter estimation for tokens requested, but pruning of tokens is not enabled; set -P to a value greater than 0\n")
        sys.exit()
    if ldf and method != "mlr":
        sys.stderr.write("Warning! Label-dependent features only work with -m mlr. Ignoring...\n")
        ldf = False
    if uniform_cost and method != "mlr":
        sys.stderr.write("Warning! Uniform cost only works with -m mlr. Ignoring...\n")
        uniform_cost = False
    if high_dim and method != "mlr":
        sys.stderr.write("Warning! High-dimensional features only work with -m 'mlr'.  Ignoring...\n")
        high_dim = False
    if matching_model > 0 and method != "cca": #currently only works with cca
        sys.stderr.write("Warning! Matching model only works with CCA.  Ignoring '-A' flag.\n'")
        matching_model = 0
    if (matching_model > 0 or source_channel) and relative_estimates and filter_cutoff == 0:
        sys.stderr.write("Warning! Filter cutoff is 0, so we don't have relative estimates information; ignorning relative estimates\n")
        relative_estimates = False
    if method == "sg" and (context_vec_loc == "" or pp_vec_loc == ""): 
        sys.stderr.write("Error! Cannot do 'sg' (skip-gram) method if context and phrase pair vector locations have not been defined! Exiting...\n")
        sys.exit()
    if source_channel and (lm_phrases_loc == "" or lm_score_loc == ""):
        sys.stderr.write("Error! With source-channel model flag on ('-d'), need to have LM scores.  Exiting...\n")
        sys.exit()
    if use_lm_filter and matching_model == 0: 
        sys.stderr.write("Warning! Matching model not enabled, but use_lm_filter enabled; ignoring the latter...\n")
        use_lm_filter = False
    if use_lm_filter and (lm_phrases_loc == "" or lm_score_loc == ""): 
        sys.stderr.write("Warning! use_lm_filter flag ('-D') enabled, but no LM information through '-a' flag provided; Ignoring...\n")
        use_lm_filter = False

    excluded_pairs = set()
    counts_dict = None
    reverse_counts_dict = None
    if filter_cutoff > 0: #if P(e|f) filtering is enabled
        counts_fh = open(filter_grammar, 'rb')
        counts_dict = cPickle.load(counts_fh)
        counts_fh.close()        
        excluded_pairs = extract_excluded_PPs(counts_dict, filter_cutoff)
        if relative_estimates:
            reverse_counts_dict = convert_counts_dict(counts_dict)

    #set up data structures and extract context features and tokens from corpus
    start = time.clock()
    tokens = None
    left_con = None
    right_con = None
    extractor = ContextExtractor(con_length, pos_depend, filter_sw, filter_features, vector_loc)
    all_phrase_pairs = set()
    if tokens_loc != "" and os.path.isfile(tokens_loc): #read-in pre-computed feature files if requested - this needs to be consistent with filtering
        tokens_fh = open(tokens_loc, 'rb')
        tokens = cPickle.load(tokens_fh)
        left_con = cPickle.load(tokens_fh)
        right_con = cPickle.load(tokens_fh)
        all_phrase_pairs = cPickle.load(tokens_fh)
        tokens_fh.close()
        print "Loaded training data from file. Time: %.1f sec"%(time.clock()-start)
    else:
        tokens = SparseContainer(prune_tokens, estimate_oov_param)
        use_wordvecs = vector_loc != ""
        left_con = SparseContext(oov) #OOV parameter is estimated by default for context
        right_con = SparseContext(oov)
        print "Sentence Count:",
        for count, line in enumerate(sys.stdin):
            source = line.strip().split(' ||| ')[0]
            phrase_pair_fh = gzip.open(phrase_pairs_loc + "/grammar.%d.gz"%count)
            extract_tokens(phrase_pair_fh, source, tokens, left_con, right_con, extractor, all_phrase_pairs, excluded_pairs, use_wordvecs)
            if count % 10000 == 0: #to track rate of feature extraction
                print "%d\t"%count,
                sys.stdout.flush()
            phrase_pair_fh.close()
        print
        print "Extracted features and phrase pairs from parallel corpus. Time: %.1f sec"%(time.clock()-start)
        tokens.create_sparse_matrix() #lose all singleton information here 
        if use_wordvecs:
            left_con.create_dense_matrix()
            right_con.create_dense_matrix()
        else:
            left_con.create_sparse_matrix(pos_depend, con_length) #take pos depend and con_length as args to make pos-dep OOV types
            right_con.create_sparse_matrix(pos_depend, con_length) #can probably simplify and remove dependence on con_length (inferred from data)
        #if not estimate_oov_param: #remove zero rows
        #    zero_rows = tokens.filter_zero_rows()
        #    left_con.filter_zero_rows(zero_rows)
        #    right_con.filter_zero_rows(zero_rows)            
        if tokens_loc != "": #write out to file if not empty
            tokens_fh = open(tokens_loc, 'wb')
            cPickle.dump(tokens, tokens_fh)
            cPickle.dump(left_con, tokens_fh)
            cPickle.dump(right_con, tokens_fh)
            cPickle.dump(all_phrase_pairs, tokens_fh)
            tokens_fh.close()
    print "Number of PP tokens: %d; Number of PP types (after all pruning): %d"%(tokens.get_token_matrix().shape[0], tokens.get_token_matrix().shape[1])
    print "Number of PPs seen (incl. singletons, excl. filtered rules): %d"%len(all_phrase_pairs)
    print "Number of excluded rules (filtered out): %d"%len(excluded_pairs)
    print "Left context dim: %d; Right context dim: %d"%(left_con.get_token_matrix().shape[1], right_con.get_token_matrix().shape[1])

    lm_scores = [] #dictionary with keys being token IDs, and value being LM score of that token
    lm_score_add = False
    if lm_score_loc != "" and (matching_model > 0 or source_channel): #read in lm_scores
        read_lm_scores(lm_phrases_loc, lm_score_loc, filter_cutoff, lm_scores)
        lm_score_add = True
        assert len(lm_scores) == tokens.get_token_matrix().shape[0]
        print "Finished reading in LM scores"
    
    heldout_idxs = None
    train_idxs = None
    if heldout_frac > 0 or shrinkage_frac > 0:
        frac_to_select = heldout_frac if heldout_frac > 0 else shrinkage_frac
        random.seed(42)
        num_tokens = tokens.get_token_matrix().shape[0]
        sample_size = int(frac_to_select*num_tokens)
        heldout_idxs = sorted(random.sample(xrange(num_tokens), sample_size))
        train_idxs = list(set(xrange(num_tokens)) - set(heldout_idxs))
        print "Selected %d out of %d samples for held-out set (some are zero rows)"%(sample_size, num_tokens)

    context = None
    left_con_train = left_con.get_token_matrix(train_idxs) if train_idxs is not None else left_con.get_token_matrix()
    right_con_train = right_con.get_token_matrix(train_idxs) if train_idxs is not None else right_con.get_token_matrix()
    if method == "sg":
        context = SGContext(context_vec_loc)
    elif context_loc != "" and os.path.isfile(context_loc): #read in pre-computed context
        context_fh = open(context_loc, 'rb')
        context = cPickle.load(context_fh)
        context_fh.close()
    else: #compute context
        context = Context(left_con_train, left_con.get_type_map(), right_con_train, right_con.get_type_map(), gamma1, rank1, whitening, mean_center, concat, con_length)
        if context_loc != "": #write out to file if not empty
            context_fh = open(context_loc, 'wb')
            cPickle.dump(context, context_fh)
            context_fh.close()
    lr_mat_l, lr_mat_r = None, None
    if method != "sg":
        lr_mat_l, lr_mat_r = context.compute_lowrank_training_contexts(left_con_train, right_con_train)
    if vector_loc != "": #word vectors take up lot of memory, so need to do some hacky garbage collection (Python sucks for this!)
        del left_con_train
        del right_con_train
        gc.collect()
    print "Computed/loaded context representations. Starting model training."

    #compute parameters for low-rank phrase disambiguation model
    model = None
    training_labels = tokens.get_token_matrix(train_idxs) if train_idxs is not None else tokens.get_token_matrix()
    if method == "cca":
        model = CCA(context, tokens.get_type_map(), all_phrase_pairs, matching_model > 0)
        if whitening == "ppmi": #can't do PPMI again because dense context representations have negative values
            whitening = "full"
        if not source_channel:
            model.train(lr_mat_l, lr_mat_r, training_labels, gamma2, rank2, whitening, mean_center)
            if matching_model > 0:
                lm_scores_train = None
                if lm_score_add and train_idxs is not None:
                    lm_scores_train = [lm_scores[idx] for idx in train_idxs]
                elif lm_score_add:
                    lm_scores_train = lm_scores
                model.train_matching_model(lr_mat_l, lr_mat_r, training_labels, matching_model, tokens.id_type_map, os.path.dirname(output_loc), lm_scores_train, use_lm_filter, counts_dict, reverse_counts_dict)
    elif method == "glm":
        model = GLM(context, tokens.get_type_map(), all_phrase_pairs, lm_score_add)
        model.train(lr_mat_l, lr_mat_r, training_labels, gamma2, lm_scores_train)
    elif method == "mlr": #multilcass logistic regression
        model = MLR(context, tokens.get_type_map(), all_phrase_pairs, ldf, high_dim)
        if high_dim:
            model.train(left_con_train, right_con_train, training_labels, gamma2, tokens.id_type_map, os.path.dirname(output_loc), uniform_cost)
        else:
            model.train(lr_mat_l, lr_mat_r, training_labels, gamma2, tokens.id_type_map, os.path.dirname(output_loc), uniform_cost)
    elif method == "mlp":
        model = MLP(context, tokens.get_type_map(), all_phrase_pairs, gamma2, rank2)
        model.train(lr_mat_l, lr_mat_r, training_labels)
    elif method == "svm":
        model = SVM(context, tokens.get_type_map(), all_phrase_pairs)
        model.train(lr_mat_l, lr_mat_r, training_labels, gamma2, os.path.dirname(output_loc))
    elif method == "sg": #skip-gram
        model = SG(context, tokens.get_type_map(), all_phrase_pairs)        
        model.train(pp_vec_loc)
    else:
        sys.stderr.write("Model argument not recognized; please input one of 'cca', 'glm', or 'mlp'\n")
        sys.exit()

    if heldout_frac > 0 or shrinkage_frac > 0: #evaluate held-out if requested
        print "Starting heldout evaluation"
        lm_scores_heldout = [lm_scores[idx] for idx in heldout_idxs] if lm_score_add else None
        sc_model = SourceChannel(lm_scores_heldout, model.type_id_map, counts_dict, reverse_counts_dict) if source_channel else None
        heldout_data = None
        if method == "sg":
            left_words, right_words = context.convert_token_matrices(left_con.get_type_map(), left_con.get_token_matrix(heldout_idxs), right_con.get_type_map(), right_con.get_token_matrix(heldout_idxs)) #returns words
            left_reps = np.array([context.get_representation(words) for words in left_words])
            right_reps = np.array([context.get_representation(words) for words in right_words])
            heldout_data = left_reps + right_reps #added together
        else:
            lr_mat_l, lr_mat_r = context.compute_lowrank_training_contexts(left_con.get_token_matrix(heldout_idxs), right_con.get_token_matrix(heldout_idxs))
            assert lr_mat_l.shape[0] == lr_mat_r.shape[0] == len(heldout_idxs)
            heldout_data = np.concatenate((lr_mat_l, lr_mat_r), axis=1)
        score_heldout(heldout_data, tokens, heldout_idxs, model, sc_model)
        #score_heldout(heldout_data, tokens, heldout_idxs, model, sc_model, lm_scores_heldout, counts_dict, reverse_counts_dict)
        if shrinkage_frac > 0:
            model.shrink_estimates(lr_mat_l, lr_mat_r, tokens, heldout_idxs)
            score_heldout(lr_mat_l, lr_mat_r, tokens, heldout_idxs, model) #score again to see improvement
    print "Model training complete"
    
    #write model to disk
    if not source_channel:
        start = time.clock()
        out_fh = open(output_loc, 'wb')
        cPickle.dump(model, out_fh)
        cPickle.dump(extractor, out_fh) #can be very large if using word vectors
        out_fh.close()
        print "Wrote parameters to disk; time: %.1f sec"%(time.clock()-start)

if __name__ == "__main__":
    main()
