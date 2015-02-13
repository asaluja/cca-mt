#!/usr/bin/python -tt

'''
File: extract_rules.py
Author: Avneesh Saluja (avneesh@cs.cmu.edu)
Date: December 22, 2014
Description: this script takes the output of the minimal grammar extractor presented in 
Zhang et al., COLING 2008 (and implemented by the same authors), which provides normalized
decomposition trees of the sentence pairs given the word alignments in s-expression format,
and converts the trees to a list of valid, minimal phrase pairs for the sentence pairs. 
Note that the phrase pairs aren't purely minimal, because for certain rules like lexicalized
rules with non-terminals, we recursively expand the children until we have a phrase pair
without any non-terminals. 
arg0: sentence pairs, in 'source ||| target' format
arg1: bidirectional word alignments
arg2: output directory for per-sentence grammars
STDIN: normalized decomposition trees (from the output of the Zhang et al. minimal grammar extractor)
Options:
-m X: adjust the maximum length of a phrase extracted (default is 6)
-l: extract lexical translations instead of phrase pairs. STDIN is ignored in this setting. 
January 20, 2015: updated and added counts dictionary writing here (for downstream feature computation).  
To Do: instead of defined s-exprsesion parser, use nltk?
'''

import sys, commands, string, collections, gzip, re, getopt, cPickle

class sexp_parser:
    '''empty initializer'''
    def __init__(self):
        pass

    '''
    greedy method that goes from the starting point in the s-expression till whenever we find
    the closing parenthesis to balance the child
    '''        
    def findEndIdx(self, substring):
        counter = 0
        idx = 0
        while (True):
            char = substring[idx]
            if char == '(':
                counter += 1
            elif char == ')':
                counter -= 1
            if counter == 0:
                break
            idx += 1
        return idx
    
    '''
    main function for parsing; goes left to right
    input: string for s-expression
    returns: tuple: a) raw_rule, a list of items in the rule b) s-expressions for the children and c) whether there is a lexical item in this rule
    '''
    def parse(self, expression):
        idx = 0
        children = []
        raw_rule = []
        isLex = False
        while idx < len(expression):
            if (expression[idx] == '(' and idx == 0) or expression[idx] == ')' or expression[idx] == ' ': 
                idx += 1
            elif expression[idx] == '(' and idx > 0: #new child
                end_child_idx = self.findEndIdx(expression[idx:]) + idx
                children.append(expression[idx:(end_child_idx+1)])
                raw_rule.append("NT")
                idx = end_child_idx + 1
            else:
                strNum = ""
                while expression[idx] != " ": #convert string to number
                    strNum += expression[idx]
                    idx += 1
                src_idx = int(strNum)-1
                raw_rule.append(src_idx)
                isLex = True
        return raw_rule, children, isLex

class minimal_tree:
    def __init__(self, src_items, tgt_items, alignment, s_tree, parent=None):
        self.children = []
        self.parent = parent
        parser = sexp_parser()
        raw_rule, children, self.lex = parser.parse(s_tree)
        src_rule = []
        tgt_rule = []
        for s_child in children:
            child_tree = minimal_tree(src_items, tgt_items, alignment, s_child, self)
            self.children.append(child_tree)
        self.source_span, self.target_span, self.rule = self.returnSpansAndRule(raw_rule, src_items, tgt_items, alignment)

    def returnSpansAndRule(self, raw_rule, src_items, tgt_items, alignment): 
        src_idxs, tgt_idxs, src_rule, tgt_rule = ([] for i in range(4))
        NT_counter = 0
        for item in raw_rule:
            if item == "NT": #collect spans of child
                child = self.children[NT_counter]
                NT_counter += 1
                src_idxs += list(child.source_span)
                tgt_idxs += list(child.target_span)
                src_rule.append("[%d]"%NT_counter)
                tgt_rule.append(("[%d]"%NT_counter, child.target_span[0])) 
            else:
                src_idxs.append(item)
                src_rule.append(src_items[item])
                for tgt_idx in alignment[item]:
                    tgt_idxs.append(tgt_idx)
                    tgt_rule.append((tgt_items[tgt_idx], tgt_idx))
        src_sp = (min(src_idxs), max(src_idxs))
        tgt_sp = (min(tgt_idxs), max(tgt_idxs))
        tgt_rule = list(set(tgt_rule))
        tgt_rule_sorted = [tup[0] for tup in sorted(tgt_rule, key = lambda tup: tup[1])]
        rule = "%s ||| %s"%(' '.join(src_rule), ' '.join(tgt_rule_sorted))
        return src_sp, tgt_sp, rule

    def returnYield(self):
        expr = re.compile(r'\[([^]]*)\]')
        if len(self.children) == 0: #pre-terminal
            return self.rule.split(' ||| ')
        else: 
            src_rule_items, tgt_rule_items = self.rule.split(' ||| ')
            src_rule = []
            tgt_rule = []
            tgt_candidates = []
            for item in src_rule_items.split():
                isNT = expr.match(item)
                if isNT: #NT
                    NT_idx = int(isNT.group(1))
                    srcYield, tgtYield = self.children[NT_idx-1].returnYield()
                    src_rule.append(srcYield)
                    tgt_candidates.append(tgtYield)
                else:
                    src_rule.append(item)
            for item in tgt_rule_items.split():
                isNT = expr.match(item)
                if isNT:
                    NT_idx = int(isNT.group(1))
                    tgt_rule.append(tgt_candidates[NT_idx-1])
                else:
                    tgt_rule.append(item)
            return ' '.join(src_rule), ' '.join(tgt_rule)

    def printTree(self, output_fh, max_length, count_dict):
        if self.lex:
            start_idx, end_idx = self.source_span
            span_length = end_idx - start_idx + 1
            if span_length <= max_length:
                if len(self.children) == 0: #pre-terminal
                    output_fh.write("%s ||| %d-%d\n"%(self.rule, start_idx, end_idx))
                    src_phrase, tgt_phrase = self.rule.split(' ||| ')
                    add_to_count_dict(count_dict, src_phrase, tgt_phrase)
                else: #lexical item in a rule with non-terminals
                    src_rule, tgt_rule = self.returnYield()
                    output_fh.write("%s ||| %s ||| %d-%d\n"%(src_rule, tgt_rule, start_idx, end_idx))
                    add_to_count_dict(count_dict, src_rule, tgt_rule)
        for child in self.children:
            child.printTree(output_fh, max_length, count_dict)

def add_to_count_dict(counts, src_phrase, tgt_phrase):
    tgt_counts = counts[src_phrase] if src_phrase in counts else {}
    tgt_counts[tgt_phrase] = tgt_counts[tgt_phrase] + 1 if tgt_phrase in tgt_counts else 1
    counts[src_phrase] = tgt_counts

def printLexicalPairs(filehandle, alignmentDict, srcWords, tgtWords, count_dict):
    for src_idx in alignmentDict:
        translation = []
        for tgt_idx in alignmentDict[src_idx]: #to handle one-to-many alignments
            translation.append(tgtWords[tgt_idx])
        filehandle.write("%s ||| %s ||| %d-%d\n"%(srcWords[src_idx], ' '.join(translation), src_idx, src_idx))
        add_to_count_dict(count_dict, srcWords[src_idx], ' '.join(translation))

def main():
    (opts, args) = getopt.getopt(sys.argv[1:], 'lm:')    
    corpus = open(args[0], 'r').read().splitlines()
    alignments = open(args[1], 'r').read().splitlines()
    output_dir = args[2]
    counts_out = args[3]
    max_length = 5
    lex = False
    for opt in opts:
        if opt[0] == '-m':
            max_length = int(opt[1])            
        elif opt[0] == '-l':
            lex = True
    data = alignments if lex else sys.stdin
    count_dict = {}
    for counter,line in enumerate(data): 
        src, tgt = corpus[counter].split(' ||| ')
        alignment = collections.defaultdict(list)
        align_pairs = line.strip().split() if lex else alignments[counter].strip().split()
        for align_pair in align_pairs:
            k,v = tuple([int(idx) for idx in align_pair.split('-')])
            alignment[k].append(v)
        out_loc = output_dir + "/grammar.%d.gz"%(counter)
        out_fh = gzip.open(out_loc, 'wb')
        if lex:
            printLexicalPairs(out_fh, alignment, src.split(), tgt.split(), count_dict)
        else:
            derivation_tree = minimal_tree(src.split(), tgt.split(), alignment, line.rstrip().lstrip())
            derivation_tree.printTree(out_fh, max_length, count_dict)
        out_fh.close()
    count_fh = open(counts_out, 'wb')
    cPickle.dump(count_dict, count_fh)
    count_fh.close()

if __name__ == "__main__":
    main()
