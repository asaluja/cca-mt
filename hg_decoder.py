#!/usr/bin/python -tt

'''
File: intersect_scfg.py
Date: December 13, 2013
Description: modified version of bottom-up parser
where we use a variant of CKY+, leveraging a trie 
data structure of the grammar to come up with a hypergraph
representation of the composition of the input sentence
FST and the grammar. The hypergraph is then converted
to a grammar where NTs are indexed/decorated with their span.
The grammar is written out in .gz files separated by sentence. 
Downstream, io.py must read in and process the per-sentence grammar
into a chart, after which the alpha and beta terms can be computed easily. 
arg1: dictionary of parameters (output of feature extraction step)
stdin: tokenized sentences
Update: incorporated simple multicore setup.  Basically, we divide the input
test corpus into partitions or chunks, and each process handles one partition. 
Usage: python intersect_scfg.py (-d/f/n) SpectralParams SentencesToDecode NumPartitions Partition Rank OutDir
Update: incorporated python's multiprocessing setup, much more efficient than the
previous multiprocessing setup.  
Usage: python intersect_scfg.py (-d/-f/-n) SpectralParams Rank InputFile NumProcesses outDir/
Update (May 20,m 2014): renamed the directory from 'inside-outside' to 'parser', and also 
renamed this file from 'intersect_scfg.py' to 'compute_hg.py'.  Changed writing out of the marginals,
instead of log marginals previously we have raw marginals.  Also, there is now an option to write
out multiple spectral marginals (normalized by source or target). 
Lastly, included an additional option flag to generate heat maps of the parse chart. 
Usage: python compute_hg.py (-d/-f/-n/-m/-s/-t) params rank input_sentences numProc outDir/
'''

import sys, commands, string, time, gzip, cPickle, re, getopt, math
import multiprocessing as mp
import numpy as np
from trie import trie, ActiveItem, HyperGraph
from eigentype import *

'''
do these as global variables because we want to share them amongst processes
if we pass them to the threads, it makes things much slower. 
'''
args = sys.argv[1:]
paramDict = cPickle.load(open(args[0], 'rb'))
output_dir = args[1]
num_process = int(args[2])
left_con = paramDict["left_con"]
right_con = paramDict["right_con"]
lowrank_con = paramDict["lowrank_con"]
tokens = paramDict["tokens"]
con_length = paramDict["con_length"]
pos_depend = paramDict["pos_depend"]
phrase_pairs = ["[X] ||| " + pair for pair in tokens.get_tokens()]
phrase_pairs.append("[X] ||| [X,1] [X,2] ||| [1] [2]")
phrase_pairs.append("[X] ||| [X,1] [X,2] ||| [2] [1]")
grammar_trie = trie(phrase_pairs)
extractor = context_extractor(con_length, pos_depend)

def format_phrase_pairs(phrase_pairs):
    phrase_dict = {}
    for phrase_pair in phrase_pairs:
        elements = phrase_pair.split(' ||| ')
        src = elements[1]
        tgt = elements[2]
        translations = phrase_dict[src] if src in phrase_dict else []
        translations.append(tgt)
        phrase_dict[src] = translations
    return phrase_dict
inventory = format_phrase_pairs(phrase_pairs)

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
        #out_fh = gzip.open(output_dir + "/grammar.%d.gz"%sent_num, 'wb')
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
            if edge.rule == "<unk>":
                out_fh.write("%s ||| <unk> ||| %s ||| cca_off=1\n"%(LHS, words[left_idx]))
            else:
                applicable_rules = inventory[edge.rule]
                left_con_words, right_con_words = extractor.extract_context(words, left_idx, right_idx-1)
                left_con_lr = left_con.get_representation(left_con_words) #closest context is last
                right_con_lr = right_con.get_representation(right_con_words) #closest context is first
                if left_con_lr is not None and right_con_lr is not None:
                    concat_con_lr = np.concatenate((left_con_lr, right_con_lr), axis=1)
                    bidi_con_lr = lowrank_con.project(concat_con_lr)
                    scored_pps = []
                    for target_phrase in applicable_rules:
                        phrase_pair = ' ||| '.join([edge.rule, target_phrase])
                        representation = tokens.get_representation([phrase_pair])
                        score = representation.dot(bidi_con_lr.transpose()) / (np.linalg.norm(representation)*np.linalg.norm(bidi_con_lr))
                        scored_pps.append((phrase_pair, score))
                    sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True)
                    for pp, score in sorted_pps:
                        out_fh.write("%s ||| %s ||| cca_on=1 cca_score=%.3f\n"%(LHS, pp, score))
                else:
                    left_null = left_con_lr is None
                    null_context_side = "left" if left_null else "right"
                    null_context = ' '.join(left_con_words) if left_null else ' '.join(right_con_words)
                    print "Phrase: '%s'; Context on %s ('%s') is completely OOV"%(' '.join(words[left_idx:right_idx]), null_context_side, null_context)
                    for target_phrase in applicable_rules:
                        phrase_pair = ' ||| '.join([edge.rule, target_phrase])
                        out_fh.write("%s ||| %s ||| cca_off=1\n"%(LHS, phrase_pair))
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
            

    
    
    
    
    
    
    
    
    
    
        
            

            
    
    
    
    
    
    
