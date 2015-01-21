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

'''
do these as global variables because we want to share them amongst processes
if we pass them to the threads, it makes things much slower. 
'''
(opts, args) = getopt.getopt(sys.argv[1:], 'n')
normalize = False
for opt in opts:
    if opt[0] == '-n':
        normalize = True
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
grammar_trie = trie(phrase_pairs)
print "Data structures from training stage loaded"

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
        print "SUCCESS; length: %d words, time taken: %.2f sec, sentence ID: %d"%(len(words), parseTime, lineNum)
        start = time.clock()
        compute_scores(hg, words, out_filename)
        cca_time = time.clock() - start
        print "Time taken to compute CCA scores: %.2f sec, sentence ID: %d"%(cca_time, lineNum)
    else:
        print "FAIL; length: %d words, time taken: %.2f sec, sentence ID: %d; sentence: %s"%(len(words), parseTime, lineNum, ' '.join(words))
        failed_sentences.append(lineNum)
    sys.stdout.flush()

def compute_scores(hg, words, out_filename):
    out_fh = gzip.open(out_filename, 'wb')
    for edge in hg.edges_:
        head = hg.nodes_[edge.headNode]
        left_idx = head.i
        right_idx = head.j
        LHS = head.cat[:-1] + "_%d_%d]"%(left_idx, right_idx)
        if len(edge.tailNodes) == 0: #pre-terminal
            if edge.rule == "<unk>": #this is <unk> in the phrase pair sense, not <unk> in the context sense
                out_fh.write("%s ||| <unk> ||| %s ||| PassThrough=1\n"%(LHS, words[left_idx]))
            else:
                left_con_words, right_con_words = extractor.extract_context(words, left_idx, right_idx-1)
                left_con_lr, right_con_lr = model.get_context_representation(left_con_words, right_con_words)
                if left_con_lr is not None and right_con_lr is not None:
                    concat_con_lr = np.concatenate((left_con_lr, right_con_lr), axis=1)
                    scored_pps = model.score(concat_con_lr, edge.rule)
                    if normalize:
                        normalizer = sum([score for pp,score in scored_pps])
                        scored_pps = [(pp, score/normalizer) for pp,score in scored_pps]
                    sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True)
                    for pp, score in sorted_pps:
                        out_fh.write("%s ||| %s ||| cca_score=%.3f\n"%(LHS, pp, score))
                else: #this occurs if all context words are stop words
                    left_null = left_con_lr is None
                    null_context_side = "left" if left_null else "right"
                    null_context = ' '.join(left_con_words) if left_null else ' '.join(right_con_words)
                    print "WARNING: Phrase: '%s'; Context on %s ('%s') was filtered; either all context words are stop words.\n"%(' '.join(words[left_idx:right_idx]), null_context_side, null_context)
                    for target_phrase in applicable_rules: #is this a good thing to do by default? 
                        phrase_pair = ' ||| '.join([edge.rule, target_phrase])
                        out_fh.write("%s ||| %s ||| cca_score=0\n"%(LHS, phrase_pair))
                if len(edge.rule.split()) == 1: #unigram, so write pass-through to be compatible with cdec
                    out_fh.write("%s ||| %s ||| %s ||| PassThrough=1\n"%(LHS, edge.rule, edge.rule))
        else: #ITG rules
            src_decorated = decorate_src_rule(hg, edge.id)
            monotone = "[1] [2] ||| Glue=1"
            inverse = "[2] [1] ||| Glue=1 Inverse=1"
            out_fh.write("%s ||| %s ||| %s\n"%(LHS, src_decorated, monotone))
            out_fh.write("%s ||| %s ||| %s\n"%(LHS, src_decorated, inverse))
    out_fh.write("[S] ||| [X_0_%d] ||| [1] ||| 0\n"%len(words)) #top level rule
    out_fh.close()

def decorate_src_rule(hg, inEdgeID):
    expr = re.compile(r'\[([^]]*)\]')
    rule = hg.edges_[inEdgeID].rule
    tail = hg.edges_[inEdgeID].tailNodes[:]
    rule_decorated = []
    for item in rule.split():
        if expr.match(item): #NT, we need to decorate with its span
            child = hg.nodes_[tail.pop(0)]
            NT = child.cat[:-1] + "_%d_%d]"%(child.i,child.j)
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
            

    
    
    
    
    
    
    
    
    
    
        
            

            
    
    
    
    
    
    

