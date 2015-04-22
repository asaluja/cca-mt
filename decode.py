#!/usr/bin/python -tt

'''
File: hg_decoder.py
Date: January 8, 2015
Author: Avneesh Saluja (avneesh@cs.cmu.edu)
Based on compute_hg.py code in spectral-scfg package
Description: modified version of bottom-up parser
where we use a variant of CKY+, leveraging a trie 
data structure of the grammar to come up with a hypergraph
representation of the composition of the input sentence
FST and the grammar. The hypergraph is then converted
to a grammar where NTs are indexed/decorated with their span.
The grammar is written out in .gz files separated by sentence. 
arg0: dictionary of parameters (output of train.py step)
arg1: output directory for per-sentence grammars
arg2: number of processes to use for decoding
'''

import sys, commands, string, time, gzip, re, getopt, math, cPickle
import multiprocessing as mp
import numpy as np
from trie import trie, ActiveItem, HyperGraph
from sparsecontainer import ContextExtractor
from model import *

def compute_signdep_means(training):
    dimensions = training.shape[1]
    pos_means = []
    neg_means = []
    for col_idx in range(dimensions):
        data = training[:,col_idx]
        pos_vals = data[data > 0]
        pos_avg = np.sum(pos_vals) / len(pos_vals) if len(pos_vals) > 0 else 0.0
        pos_means.append(pos_avg)
        neg_vals = data[data < 0]
        neg_avg = np.sum(neg_vals) / len(neg_vals) if len(neg_vals) > 0 else 0.0
        neg_means.append(neg_avg)
    return pos_means, neg_means

def compute_feature_thresholds(model, tokens_loc):
    tokens_fh = open(tokens_loc, 'rb')
    tokens = cPickle.load(tokens_fh)
    left_con = cPickle.load(tokens_fh)
    right_con = cPickle.load(tokens_fh)
    tokens_fh.close()
    lr_mat_l, lr_mat_r = model.context.compute_lowrank_training_contexts(left_con.get_token_matrix(), right_con.get_token_matrix())
    training_data = np.concatenate((lr_mat_l, lr_mat_r), axis=1)
    context_pos_means, context_neg_means = None, None
    try:
        lowdim_context = training_data.dot(model.context_parameters)
        context_pos_means, context_neg_means = compute_signdep_means(lowdim_context)
    except AttributeError:
        context_pos_means, context_neg_means = compute_signdep_means(training_data)
    model.set_context_means(context_pos_means, context_neg_means)
    if not model.isvw(): #for CCA and GLM models, where parameters are transparent
        lowdim_pps = tokens.get_token_matrix().dot(model.parameters.transpose())
        pp_pos_means, pp_neg_means = compute_signdep_means(lowdim_pps)
        model.set_phraserep_means(pp_pos_means, pp_neg_means)

'''
do these as global variables because we want to share them amongst processes
if we pass them to the threads, it makes things much slower. 
'''
(opts, args) = getopt.getopt(sys.argv[1:], 'bd:cCln:oprR')
normalize = "none"
represent = False
discretize = ""
covariance = False
plain = False
no_cca = False
log_score = False
only_pos = False
print_best = False
print_rank = False
for opt in opts:
    if opt[0] == '-n':
        normalize = opt[1]
    elif opt[0] == '-r':
        represent = True
    elif opt[0] == '-d': 
        discretize = opt[1] #location of sparse tokens and context
    elif opt[0] == '-p': #plain - no markup
        plain = True
    elif opt[0] == '-c': #no cca_on/off feature
        no_cca = True
    elif opt[0] == '-C': #covariance/second order feature for vectors
        covariance = True
    elif opt[0] == '-l': #put scores in log space
        log_score = True
    elif opt[0] == '-o': #only positive scores (0-1)
        only_pos = True
    elif opt[0] == '-b': #print best
        print_best = True
    elif opt[0] == '-R': #print rak
        print_rank = True
if normalize != "none" and normalize != "exp" and normalize != "range":
    sys.stderr.write("Error! normalization option not recognized (valid options are 'none', 'exp', and 'range'). Setting to 'none'\n")
    normalize = "none"
if discretize and not represent: 
    sys.stderr.write("Error! Cannot have discretize on ('-d') without representations being printed out ('-r'); Turning it off\n")
    discretize = False
if covariance and not represent:
    sys.stderr.write("Error! Cannot have covariance features on ('-C') without representations being printed out ('-r'); Turning it off\n")
    covariance = False
if not only_pos and log_score: #would result in domain errors
    sys.stderr.write("Error! Cannot have log score without restricting scores to be positive; disabling log score\n")
    log_score = False
if log_score and no_cca: #if we are not writing scores, then log scores will be ignored
    sys.stderr.write("Warning! Ignoring log_score ('-l') option, since no_cca flag ('-c') is on\n")    

param_filename = args[0]
output_dir = args[1]
num_process = int(args[2])
param_fh = open(param_filename, 'rb')
model = cPickle.load(param_fh)
extractor = cPickle.load(param_fh)
param_fh.close()
phrase_pairs = ["[X] ||| " + pair for pair in model.get_tokens()]
phrase_pairs.append("[X] ||| [X,1] [X,2] ||| [1] [2]")
phrase_pairs.append("[X] ||| [X,1] [X,2] ||| [2] [1]")
#dev_grammars=args[3]
grammar_trie = trie(phrase_pairs)
print "Data structures from training stage loaded"
if discretize != "": #compute relevant statistics for discretization
    compute_feature_thresholds(model, discretize)

'''
declaration of list that maintains which sentences have failed across all processes
'''
def init(fs):
    global failed_sentences
    failed_sentences = fs

def main():
    failed_sentences = mp.Manager().list()
    pool = mp.Pool(processes=num_process, initializer=init, initargs=(failed_sentences,))
    for sent_num, line in enumerate(sys.stdin):
        out_filename = output_dir + "/grammar.%d.gz"%sent_num
        #parse(line.strip().split(), out_filename, sent_num)
        pool.apply_async(parse, (line.strip().split(), out_filename, sent_num))
    pool.close()
    pool.join()
    print "number of failed sentences: %d"%(len(failed_sentences))

'''
main function for bottom-up parser with Earley-style rules. 
The active chart is first seeded with pointers to the root
node of a source rules trie. Then, in a bottom-up manner, 
we advance the dots for each cell item, and then convert completed
rules in a cell to the passive chart, or deal with NTs in active
items just proved.  At the end, we look at the passive items in
the cell corresponding to the sentence to see if [S] is there. 
'''
def parse(words, out_filename, lineNum):
    start = time.clock()
    N = len(words)
    goal_idx = False
    hg = HyperGraph()
    active = {}
    passive = {}
    nodemap = {}
    seedActiveChart(N, active)
    dev_rules = {}
    #dev psg read in (for oracle)
    #dev_filename = dev_grammars + "/grammar.%d.gz"%lineNum
    #dev_fh = gzip.open(dev_filename, 'rb')
    #for line in dev_fh:
    #    src,tgt,align = line.strip().split(' ||| ')
    #    key = (src, align)
    #    dev_rules[key] = tgt
    #dev_fh.close()
    for l in range(1, N+1): #length
        for i in range(0, N+1-l): #left index of span            
            j = i + l #right index of span
            advanceDotsForAllItemsInCell(i, j, words, active, passive)
            cell = active[(i,j)][:] if (i,j) in active else [] #list of active items
            for activeItem in cell:
                rules = activeItem.srcTrie.getRules()
                for rule in rules:
                    applyRule(i, j, rule, activeItem.tailNodeVec, hg, nodemap, passive)
            if j < N: #the below function includes NTs that were just proved into new binaries, which is unnecessary for the end token
                extendActiveItems(i, i, j, active, passive) #dealing with NTs that were just proved
        if (0,N) in passive: #we have spanned the entire input sentence
            passiveItems = passive[(0,N)] #list of indices
            if len(passiveItems) > 0: #we have at least one node that covers the entire sentence
                goal_idx = True                
    parseTime = time.clock() - start
    if goal_idx: #i.e., if we have created at least 1 node in the HG corresponding to goal        
        print "Parsed sentence; length: %d words, time taken: %.2f sec, sentence ID: %d"%(len(words), parseTime, lineNum)
        start = time.clock()
        compute_scores(hg, words, out_filename, dev_rules)
        cca_time = time.clock() - start
        print "SUCESS! Time taken to compute scores: %.2f sec, sentence ID: %d"%(cca_time, lineNum)
    else:
        print "FAIL; length: %d words, time taken: %.2f sec, sentence ID: %d; sentence: %s"%(len(words), parseTime, lineNum, ' '.join(words))
        failed_sentences.append(lineNum)
    sys.stdout.flush()

def compute_scores(hg, words, out_filename, dev_rules):
    rules_out = []
    phrases_to_score = []
    phrases_for_oracle = [] #this has been added
    for edge in hg.edges_: #first, go through hypergraph process rules that can be written out; store scorable rules for scoring later
        head = hg.nodes_[edge.headNode]
        left_idx = head.i
        right_idx = head.j
        LHS = head.cat[:-1] + "_%d_%d]"%(left_idx, right_idx) if not plain else head.cat 
        if len(edge.tailNodes) > 0: #ITG rules
            src_decorated = decorate_src_rule(hg, edge.id)
            monotone = "[1] [2] ||| Glue=1"
            inverse = "[2] [1] ||| Glue=1 Inverse=1"
            rules_out.append("%s ||| %s ||| %s"%(LHS, src_decorated, monotone))
            rules_out.append("%s ||| %s ||| %s"%(LHS, src_decorated, inverse))
        else:
            if edge.rule == "<unk>": #phrase is not in translation inventory
                rules_out.append("%s ||| <unk> ||| %s ||| PassThrough=1"%(LHS, words[left_idx]))
            else:
                left_con_words, right_con_words = extractor.extract_context(words, left_idx, right_idx-1) #same as before, either real-valued arrays or lists of words
                left_con_lr, right_con_lr = model.get_context_rep_vec(left_con_words, right_con_words) if extractor.is_repvec() else model.get_context_rep(left_con_words, right_con_words)
                if len(edge.rule.split()) == 1: #unigram, so write pass-through to be compatible with cdec
                    rules_out.append("%s ||| %s ||| %s ||| PassThrough=1"%(LHS, edge.rule, edge.rule))
                if left_con_lr is not None and right_con_lr is not None: #valid context
                    concat_con_lr = np.concatenate((left_con_lr, right_con_lr))
                    phrases_to_score.append((LHS, edge.rule, concat_con_lr))
                    phrases_for_oracle.append((edge.rule, left_idx, right_idx-1)) #this has been added
                else: #this occurs if all context words are stop words - may want to edit what happens in this branch condition in light of recent changes
                    left_null = left_con_lr is None
                    null_context_side = "left" if left_null else "right"
                    null_context = ' '.join(left_con_words) if left_null else ' '.join(right_con_words)
                    print "WARNING: Phrase: '%s'; Context on %s ('%s') was filtered; all context words are stop words.\n"%(' '.join(words[left_idx:right_idx]), null_context_side, null_context)
                    for target_phrase in applicable_rules: #is this a good thing to do by default? 
                        phrase_pair = ' ||| '.join([edge.rule, target_phrase])
                        rules_out.append("%s ||| %s ||| cca_on=1 cca_score=0"%(LHS, phrase_pair))
    scored_pps_all = []
    if model.isvw(): #score all phrases in sentence together
        if len(phrases_to_score) > 0:
            LHS, src_phrases, context_reps = zip(*phrases_to_score)
            context_reps = np.vstack(context_reps)
            sent_num = int(out_filename.split("/")[-1].split('.')[1])
            scored_pps_all = model.score_all(context_reps, src_phrases, sent_num, represent)
    else:
        for LHS, src_phrase, context_rep in phrases_to_score:
            scored_pps = model.score(context_rep, src_phrase, represent) #scored_pps gives idx
            scored_pps_all.append(scored_pps) 
    for idx,pps_to_score in enumerate(scored_pps_all): #final processing of scored phrases
        LHS = phrases_to_score[idx][0]
        #src_phrase, left_idx, right_idx_incl = phrases_for_oracle[idx]
        #key = (src_phrase, "%d-%d"%(left_idx, right_idx_incl))
        #source_oracle = True if key in dev_rules else False
        #target_oracle = dev_rules[key] if key in dev_rules else ""
        scored_pps = []
        if normalize == "exp":
            scored_pps = normalize_exp(pps_to_score)
        elif normalize == "range":
            scored_pps = normalize_range(pps_to_score, only_pos, log_score)
        else:
            if only_pos:
                for pp,score,reps in pps_to_score:
                    if score is None:
                        scored_pps.append((pp, score, reps))
                    else:
                        scored_pps.append((pp, 1-np.arccos(score)/math.pi, reps))
            else:
                scored_pps = pps_to_score
        sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True)
        best_pp = sorted_pps[0][0]
        rank = 0
        for pp, score, rep in sorted_pps:
            rank += 1
            rule_str = "%s ||| %s ||| "%(LHS, pp)
            if not no_cca: #meaning we can decorate
                rule_str += "cca_off=1" if score is None else "cca_on=1"
            if not plain and score is not None:
                if log_score:
                    score = -math.log10(score)
                elif print_rank:
                    score = rank
                rule_str += " cca_score=%.3f"%score if log_score else " cca_score=%.5g"%score
                if print_best and pp == best_pp: 
                    rule_str += " cca_best=1"
            if represent: #if outputting context and/or phrase pair representations
                assert rep != ""      
                if covariance:
                    rep = compute_second_order(rep); 
                rule_str += " %s"%rep 
            #if source_oracle:
            #    rule_str += " oracle_source=1"
            #target = pp.split(' ||| ')[1]
            #if target == target_oracle:
            #    rule_str += " oracle_target=1"
            rules_out.append(rule_str)
    rules_out = list(set(rules_out)) #makes rules unique
    out_fh = gzip.open(out_filename, 'wb')
    for rule in rules_out:
        out_fh.write("%s\n"%rule)
    top_rule = "[S] ||| [X_0_%d] ||| [1] ||| 0\n"%len(words) if not plain else "[S] ||| [X] ||| [1] ||| 0\n"
    out_fh.write(top_rule)
    out_fh.close()

def compute_second_order(rep):
    str_rep = []
    context_vals = np.array([float(element.split('=')[1]) for element in rep.split() if element.split('=')[0].split('_')[0] == 'c'])
    pp_vals = np.array([float(element.split('=')[1]) for element in rep.split() if element.split('=')[0].split('_')[0] == 'pp'])
    outer_prod = np.outer(context_vals, pp_vals)
    for idx,val in enumerate(outer_prod.flatten()): 
        str_rep.append("op%d=%.3g"%(idx,val))
        #str_rep.append("op_dim%d=%.3g"%(idx,val))
    return ' '.join(str_rep)

'''
no need to pass only_pos to normalize_exp because result of exp normalization is always positive
'''
def normalize_exp(scored_pps):
    normalizer = sum([math.exp(score) for pp,score,reps in scored_pps if score is not None]) #will be zero if all is none
    if normalizer != 0:
        normalizer = math.log(normalizer)
    normalized_pps = []
    for pp, score, reps in scored_pps:
        if score is None:
            normalized_pps.append((pp, score, reps))
        else: #normalizer can be 0 if before log it was 1 (meaning raw score was zero)
            pp_norm = (pp, math.exp(score-normalizer), reps) #equivalent to exp(score) / sum_i exp(score_i)
            normalized_pps.append(pp_norm)
    return normalized_pps

def normalize_range(scored_pps, only_pos, log_score):
    sum_vals = 0
    num_vals = 0
    min_val = 1000000.0
    max_val = -1000000.0
    for pp, score, reps in scored_pps: #loop through and calculate necessary statistics
        if score is not None:
            num_vals += 1
            sum_vals += score
            max_val = score if score > max_val else max_val
            min_val = score if score < min_val else min_val
    if num_vals > 0:        
        average = sum_vals / num_vals
        normalizer = max_val - min_val
        shift = (min_val - average) / normalizer if only_pos and normalizer != 0 else 0
        plus_one = 1 if log_score else 0
        normalized_pps = []
        for pp, score, reps in scored_pps:
            if score is None:
                normalized_pps.append((pp, score, reps))
            else:
                pp_norm = (pp, (score-average)/normalizer - shift + plus_one, reps) if normalizer != 0 else (pp, 1./num_vals, reps)                
                normalized_pps.append(pp_norm)
        return normalized_pps
    else:        
        return scored_pps

def decorate_src_rule(hg, inEdgeID):
    expr = re.compile(r'\[([^]]*)\]')
    rule = hg.edges_[inEdgeID].rule
    tail = hg.edges_[inEdgeID].tailNodes[:]
    rule_decorated = []
    for item in rule.split():
        if expr.match(item): #NT, we need to decorate with its span
            child = hg.nodes_[tail.pop(0)]
            NT = child.cat[:-1] + "_%d_%d]"%(child.i,child.j) if not plain else child.cat
            rule_decorated.append(NT)
        else:
            rule_decorated.append(item)
    return ' '.join(rule_decorated)

'''
Function called before the sentence is parsed;
places a pointer to the source rules trie root
along the diagonal of the active chart. 
'''
def seedActiveChart(N, active):
    for i in range(0, N): #note: for now, we don't test hasRuleForSpan        
        active.setdefault((i,i), []).append(ActiveItem(grammar_trie.getRoot())) #add the root of the trie

'''
Function that "advances the dot" (in a dotted rule)
on position to the right for all active items in the cell
defined by (start,end).  We first perform online binarization
by looping through all split points in the span and then see if
advancing the dot happened to cover a non-terminal (this is handled
in extendActiveItems).  We then check and see if advancing the dot 
happened to cover a new rule with the additional terminal.  
'''
def advanceDotsForAllItemsInCell(start, end, words, active, passive):
    for k in range(start+1, end):
        extendActiveItems(start, k, end, active, passive)        
    ec = active[(start,end-1)] if (start,end-1) in active else []
    word = words[end-1]
    for actItem in ec:
        ai = actItem.extendTerminal(word)
        if ai is not None:
            active.setdefault((start,end), []).append(ai)
        if end-start == 1: #OOV handling
            if ai is None:
                active.setdefault((start,end), []).append(actItem.extendOOV())                
            else: #check if active item has any rules in its bin
                if len(ai.srcTrie.getRules()) == 0: #handles the case where rule starts with OOV word, but no rule that actually covers OOV word
                    active.setdefault((start,end), []).append(actItem.extendOOV())

'''
function that extends active items over non-terminals. 
'''
def extendActiveItems(start, split, end, active, passive):
    icell = active[(start, split)] if (start,split) in active else []
    idxs = passive[(split, end)] if (split,end) in passive else []
    for actItem in icell:
        for idx in idxs:
            ai = actItem.extendNonTerminal(idx) 
            if ai is not None:
                active.setdefault((start,end), []).append(ai)

'''
Given a rule, does the necessary book-keeping to 
convert that rule to the passive chart, and adds the 
appropriate nodes and edges to the hypergraph. 
'''
def applyRule(start, end, rule, tailNodes, hg, nodemap, passive):
    edge = hg.addEdge(rule[1], tailNodes) #rule[1] is src RHS of rule
    node = None
    cat2NodeMap = {}
    if (start,end) in nodemap:
        cat2NodeMap = nodemap[(start,end)]    
    LHS = rule[0]
    if LHS in cat2NodeMap: #LHS is either [X] or [S] --> test if this ever fires?
        node = hg.nodes_[cat2NodeMap[LHS]]
    else:
        node = hg.addNode(LHS, start, end)
        cat2NodeMap[LHS] = node.id
        nodemap[(start,end)] = cat2NodeMap
        passive.setdefault((start,end), []).append(node.id)
    hg.connectEdgeToHeadNode(edge, node)

if __name__ == "__main__":
    main()
            

    
    
    
    
    
    
    
    
    
    
        
            

            
    
    
    
    
    
    

