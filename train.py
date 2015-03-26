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
Flags: see README.md for more details
'''

import sys, commands, string, gzip, getopt, os, cPickle, time, random, gc
from scipy.special import expit
from sparsecontainer import *
from model import *
import resource

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

def score_heldout(left_low_rank, right_low_rank, tokens, heldout_idxs, model):
    assert left_low_rank.shape[0] == right_low_rank.shape[0] == len(heldout_idxs)
    start = time.clock()    
    heldout_data = np.concatenate((left_low_rank, right_low_rank), axis=1)
    mrr = 0
    mrr_hard_examples = 0
    multi_translation_count = 0
    hard_examples_count = 0
    avg_length = 0
    phrases = []
    ground_truth = []
    example_idxs = []
    sq_loss = 0
    print "Heldout: beginning evaluation"
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
    scored_pps_all = [] #all src phrases
    if model.isvw(): #then score all phrases together
        scored_pps_all = model.score_all(heldout_data[example_idxs,:], phrases, 0, False)
    else:
        for idx,src_phrase in enumerate(phrases): #score each phrase individually
            scored_pps = model.score(heldout_data[example_idxs[idx],:], src_phrase, False)
            scored_pps_all.append(scored_pps)
    print "Heldout: scored phrases"
    for idx,scored_pps in enumerate(scored_pps_all): #compute MRR for scored PPs
        if len(scored_pps) > 1: #only record MRR for src phrases with more than one translation
            avg_length += len(scored_pps)
            sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True) #sort by score to get rank
            phrase_pair = ground_truth[idx]
            scored_rank = [rank+1 for rank, pp_score in enumerate(sorted_pps) if pp_score[0] == phrase_pair][0] #because we filtered for singleton PPs, guaranteed that scored_rank is not None
            mrr += 1./scored_rank
            multi_translation_count += 1
            #score = expit([pp_score[1] for pp_score in sorted_pps if pp_score[0] == phrase_pair][0]) #for passing through logistic function
            #sq_loss += (1-score)**2 
            if scored_rank > 1: #separate tracking if correct translation was not ranked 1
                mrr_hard_examples += 1./scored_rank
                hard_examples_count += 1
    mrr /= multi_translation_count
    #sq_loss /= multi_translation_count
    mrr_hard_examples /= hard_examples_count
    avg_length = float(avg_length) / multi_translation_count
    print "Heldout: computed MRR. Total time: %.1f sec"%(time.clock()-start)
    print "Out of %d examples (source phrases), %d are not pruned (singleton/filtered), %d have > 1 translation (restricting scoring to these phrases)"%(len(heldout_idxs), len(scored_pps_all), multi_translation_count)
    print "Mean Reciprocal Rank: %.3f; Average number of translations per phrase: %.2f"%(mrr, avg_length)
    print "Mean Reciprocal Rank for phrase pairs not ranked 1: %.3f; Number of such examples: %d"%(mrr_hard_examples, hard_examples_count)

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

def read_lm_scores(lm_scores_loc, lm_scores):
    lm_fh = open(lm_scores_loc, 'rb')
    for line in lm_fh:
        val = float(line.strip().split()[1])
        length = int(line.strip().split()[-1])
        normalized_val = val / length
        lm_scores.append(normalized_val)
    lm_fh.close()
    
def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'a:cC:f:F:g:h:Hl:Lm:Mo:OpP:s:S:r:t:uv:w:')
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
            lm_score_loc = opt[1]
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
    if not (method == "cca" or method == "glm" or method == "mlr" or method == "mlp" or method == "svm"):
        sys.stderr.write("Error! Currently supported supervised methods are 'cca', 'glm', 'mlr', 'svm', or 'mlp'\n")
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

    excluded_pairs = set()
    if filter_cutoff > 0: #if P(e|f) filtering is enabled
        counts_fh = open(filter_grammar, 'rb')
        counts_dict = cPickle.load(counts_fh)
        counts_fh.close()        
        excluded_pairs = extract_excluded_PPs(counts_dict, filter_cutoff)
    #set up data structures and extract context features and tokens from corpus
    start = time.clock()
    tokens = None
    left_con = None
    right_con = None
    extractor = ContextExtractor(con_length, pos_depend, filter_sw, filter_features, vector_loc)
    all_phrase_pairs = set()
    lm_scores = [] #dictionary with keys being token IDs, and value being LM score of that token
    lm_score_add = False
    if lm_score_loc != "": #read in lm_scores
        read_lm_scores(lm_score_loc, lm_scores)
        lm_score_add = True
    lm_scores = np.array(lm_scores)
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
    if context_loc != "" and os.path.isfile(context_loc): #read in pre-computed context
        context_fh = open(context_loc, 'rb')
        context = cPickle.load(context_fh)
        context_fh.close()
    else: #compute context
        context = Context(left_con_train, left_con.get_type_map(), right_con_train, right_con.get_type_map(), gamma1, rank1, whitening, mean_center, concat, con_length)
        if context_loc != "": #write out to file if not empty
            context_fh = open(context_loc, 'wb')
            cPickle.dump(context, context_fh)
            context_fh.close()
    lr_mat_l, lr_mat_r = context.compute_lowrank_training_contexts(left_con_train, right_con_train)
    if vector_loc != "": #word vectors take up lot of memory, so need to do some hacky garbage collection (Python sucks for this!)
        del left_con_train
        del right_con_train
        gc.collect()
    print "Computed/loaded context representations. Starting model training."

    #compute parameters for low-rank phrase disambiguation model
    model = None
    training_labels = tokens.get_token_matrix(train_idxs) if train_idxs is not None else tokens.get_token_matrix()
    lm_scores_train = lm_scores[train_idxs] if lm_score_add and train_idxs is not None else lm_scores
    if method == "cca":
        model = CCA(context, tokens.get_type_map(), all_phrase_pairs)
        if whitening == "ppmi": #can't do PPMI again because dense context representations have negative values
            whitening = "full"
        model.train(lr_mat_l, lr_mat_r, training_labels, gamma2, rank2, whitening, mean_center)
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
    else:
        sys.stderr.write("Model argument not recognized; please input one of 'cca', 'glm', or 'mlp'\n")
        sys.exit()
    if heldout_frac > 0 or shrinkage_frac > 0: #evaluate held-out if requested
        print "Starting heldout evaluation"
        lr_mat_l, lr_mat_r = context.compute_lowrank_training_contexts(left_con.get_token_matrix(heldout_idxs), right_con.get_token_matrix(heldout_idxs))    
        score_heldout(lr_mat_l, lr_mat_r, tokens, heldout_idxs, model)
        if shrinkage_frac > 0:
            model.shrink_estimates(lr_mat_l, lr_mat_r, tokens, heldout_idxs)
            score_heldout(lr_mat_l, lr_mat_r, tokens, heldout_idxs, model) #score again to see improvement    
    print "Model training complete"
    
    #write model to disk
    start = time.clock()
    out_fh = open(output_loc, 'wb')
    cPickle.dump(model, out_fh)
    cPickle.dump(extractor, out_fh) #can be very large if using word vectors
    out_fh.close()
    print "Wrote parameters to disk; time: %.1f sec"%(time.clock()-start)

if __name__ == "__main__":
    main()
