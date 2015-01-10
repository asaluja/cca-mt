#!/usr/bin/python -tt

'''
File: decoding.py
Author: Avneesh Saluja (avneesh@cs.cmu.edu)
Date: January 2, 2015
Description: currently a quick and dirty decoder for lexical translation,
where we do a k-NN based scoring. 
'''

import sys, commands, string, cPickle
from eigentype import *
import numpy as np

class trieNode:
    '''
    TrieNode class is the recursive data structure that trie
    relies on.  Each node contains a list of the rules that
    terminate at that node, and a dictionary that maps
    words (edges) to other trieNodes. 
    '''
    def __init__(self): #constructor for trieNode
        self.edges_ = {}
        self.rb_ = []
        
    def addEdge(self, item):
        if item not in self.edges_:
            self.edges_[item] = trieNode()

    def addRule(self, rule): #rule is an (LHs, src RHS) tuple - change this
        self.rb_.append(rule)
        self.rb_ = list(set(self.rb_))

    def getRules(self):
        return self.rb_
            
    #returns a trieNode
    def Extend(self, word):
        if word in self.edges_:
            return self.edges_[word]
        else:
            return None

class trie:
    '''
    Trie class is a data structure used for speeding up decoding. 
    The trie class takes as input a list of all the rules in 
    the grammar, and constructs a trie where the edges correspond
    to words and each node contains a bin of all the rules
    that terminate at that node.  Given this data structure, we can 
    quickly retrieve the rules in the grammar that match a given 
    sub-span of a sentence.  
    '''
    def __init__(self, rules): #constructor for trie
        self.root = trieNode()
        for rule in rules: #rules is a list of rules
            elements = rule.split(' ||| ')
            src = elements[0]
            #rule_src = self.formatRule(RHS_src.split()) #if rule contains NTs
            rule_src = src.split()
            curNode = self.root
            for item in rule_src: #add the rule to the trie
                curNode.addEdge(item)
                nextNode = curNode.Extend(item)
                curNode = nextNode
            curNode.addRule(rule)
        #can probably remove the below depending on how we handle in decoding
        self.root.addEdge("<unk>") #adding OOV token
        OOVNode = self.root.Extend("<unk>")
        OOVNode.addRule(("[X]","<unk>"))
    
    def formatRule(self, rule): #rule is the source phrase broken down into a list
        exp = re.compile(r'\[([^]]*)\]')
        rule_f = []
        for item in rule:
            if exp.match(item):
                rule_f.append('[X]')
            else:
                rule_f.append(item)
        return rule_f

    def getRoot(self):
        return self.root
    
    '''for debugging'''
    def traverseTrie(self, pos, curNode=None):
        if pos == 0:
            curNode = self.root
        print "position %d rule bin: "%pos
        print curNode.getRules()
        for nextNode in curNode.edges_.keys():
            print "Traversing edge corresponding to %s"%(nextNode)
            self.traverseTrie(pos+1, curNode.Extend(nextNode))
            print "Returning to previous node"

def format_phrase_pairs(phrase_pairs):
    phrase_dict = {}
    for phrase_pair in phrase_pairs:
        elements = phrase_pair.split(' ||| ')
        src = elements[0]
        tgt = elements[1]
        translations = phrase_dict[src] if src in phrase_dict else []
        translations.append(tgt)
        phrase_dict[src] = translations
    return phrase_dict

'''
faster implementation that makes use of trie data structure to make the 
algorithm output-insensitive
'''
def translate_line_with_trie(sentence, inventory, extractor, left_con, right_con, lowrank_con, tokens, out_fh):
    words = sentence.split()
    for left_idx in range(len(words)):
        cur_node = inventory.getRoot()
        right_idx = left_idx
        while right_idx < len(words):
            next_node = cur_node.Extend(words[right_idx])
            if next_node is not None:
                cur_node = next_node
                applicable_rules = cur_node.getRules() #list of phrase pairs --> this actually may be empty
                left_con_words, right_con_words = extractor.extract_context(words, left_idx, right_idx)
                left_con_lr = left_con.get_representation(left_con_words) #closest context is last
                right_con_lr = right_con.get_representation(right_con_words) #closest context is first
                if left_con_lr is not None and right_con_lr is not None:
                    concat_con_lr = np.concatenate((left_con_lr, right_con_lr), axis=1)
                    bidi_con_lr = lowrank_con.project(concat_con_lr)
                    scored_pps = []
                    for rule in applicable_rules:
                        representation = tokens.get_representation([rule])
                        score = representation.dot(bidi_con_lr.transpose()) / (np.linalg.norm(representation)*np.linalg.norm(bidi_con_lr))
                        scored_pps.append((rule, score))                        
                    sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True)
                    for pp, score in sorted_pps:
                        #out_fh.write("[X_%d_%d] ||| %s ||| cca_on=1 cca_score=%.3f\n"%(left_idx, right_idx+1, pp, score))
                        out_fh.write("[X] ||| %s ||| cca_on=1 cca_score=%.3f\n"%(pp, score))
                else:
                    left_null = left_con_lr is None
                    null_context_side = "left" if left_null else "right"
                    null_context = ' '.join(left_con_words) if left_null else ' '.join(right_con_words)
                    print "Phrase: '%s'; Context on %s ('%s') is completely OOV"%(' '.join(words[left_idx:right_idx+1]), null_context_side, null_context)
                    for rule in applicable_rules:
                        #out_fh.write("[X_%d_%d] ||| %s ||| cca_off=1\n"%(left_idx, right_idx+1, rule))
                        out_fh.write("[X] ||| %s ||| cca_off=1\n"%(rule))
                right_idx += 1
            else: #need to check if next_node is not None because of OOV or not
                if right_idx - left_idx == 0: #unigram OOV
                    print "Word '%s' is OOV"%words[left_idx]
                    #out_fh.write("[X_%d_%d] ||| <unk> ||| %s ||| cca_off=1\n"%(left_idx, right_idx+1, words[left_idx]))
                    out_fh.write("[X] ||| <unk> ||| %s ||| cca_off=1\n"%(words[left_idx]))
                break
    out_fh.close()
    
'''
general implementation that works for phrase-based translation and lexical translation; 
O(n^2) complexity, where n is the length of the sentence
'''
def translate_line(sentence, inventory, extractor, left_con, right_con, lowrank_con, tokens, out_fh):
    words = sentence.split()
    for left_idx in range(len(words)):
        for right_idx in range(left_idx, len(words)):
            right_idx += 1
            phrase = ' '.join(words[left_idx:right_idx])
            if phrase in inventory: #we have seen this source phrase in training before and have at least one translation
                translations = inventory[phrase]
                representations = []
                for translation in translations: #in this loop, don't need to store the entire phrase_pair since src_phrase is given
                    phrase_pair = ' ||| '.join([phrase, translation])
                    representations.append((phrase_pair, tokens.get_representation([phrase_pair]))) #guaranteed that get_representation returns a non-NULL vector, because phrase is in inventory
                left_con_words, right_con_words = extractor.extract_context(words, left_idx, right_idx)
                left_con_lr = left_con.get_representation(left_con_words) #closest context is last
                right_con_lr = right_con.get_representation(right_con_words) #closest context is first
                if left_con_lr is not None and right_con_lr is not None:
                    concat_con_lr = np.concatenate((left_con_lr, right_con_lr), axis=1)
                    bidi_con_lr = lowrank_con.project(concat_con_lr)
                    scored_pps = []
                    for phrase_pair, representation in representations:
                        score = representation.dot(bidi_con_lr.transpose()) / (np.linalg.norm(representation)*np.linalg.norm(bidi_con_lr))
                        scored_pps.append((phrase_pair, score))
                    sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True)
                    for pp, score in sorted_pps:
                        out_fh.write("%s ||| %s ||| cca_on=1 cca=%.3f\n"%(phrase, pp.split(' ||| ')[1], score))
                else:
                    left_null = left_con_lr is None
                    null_context_side = "left" if left_null else "right"
                    null_context = ' '.join(left_con_words) if left_null else ' '.join(right_con_words)
                    print "Phrase: '%s'; Context on %s ('%s') is completely OOV"%(phrase, null_context_side, null_context)
                    for phrase_pair, representation in representations:
                        out_fh.write("%s ||| %s ||| cca_off=1\n"%(phrase, phrase_pair.split(' ||| ')[1]))
            else:
                print "Phrase '%s' is OOV"%phrase
                if len(phrase.split()) == 1: #unigram
                    out_fh.write("%s ||| %s ||| cca_off=1\n"%("<unk>", phrase))
    out_fh.close()
                                     
def main():
    args = sys.argv[1:]
    paramDict = cPickle.load(open(args[0], 'rb'))
    output_dir = args[1]
    left_con = paramDict["left_con"]
    right_con = paramDict["right_con"]
    bidi_lowrank_con = paramDict["lowrank_con"]
    tokens = paramDict["tokens"]
    con_length = paramDict["con_length"]
    pos_depend = paramDict["pos_depend"]
    #inventory = format_phrase_pairs(tokens.get_tokens())
    inventory = trie(tokens.get_tokens())
    for sent_num, line in enumerate(sys.stdin):
        extractor = context_extractor(con_length, pos_depend)
        out_fh = gzip.open(output_dir + "/grammar.%d.gz"%sent_num, 'wb')
        #translate_line(line.strip(), inventory, extractor, left_con, right_con, bidi_lowrank_con, tokens, out_fh)
        translate_line_with_trie(line.strip(), inventory, extractor, left_con, right_con, bidi_lowrank_con, tokens, out_fh)

if __name__ == "__main__":
    main()
