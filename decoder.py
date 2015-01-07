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
general implementation that works for phrase-based translation and lexical translation; 
O(n^2) complexity, where n is the length of the sentence
'''
def translate_line(sentence, inventory, extractor, left_con, right_con, lowrank_con, tokens, out_fh):
    words = sentence.split()
    for left_idx in range(len(words)):
        for right_idx in range(left_idx+1, len(words)):
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
                        out_fh.write("%s ||| %s ||| cca_off=1"%(phrase, pp.split(' ||| ')[1]))
            else:
                print "Phrase '%s' is OOV"%phrase
                if len(phrase.split()) == 1: #unigram
                    out_fh.write("%s ||| %s ||| cca_off=1"%("<unk>", phrase))
                                     
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
    inventory = format_phrase_pairs(tokens.get_tokens())
    for sent_num, line in enumerate(sys.stdin):
        extractor = context_extractor(con_length, pos_depend)
        out_fh = gzip.open(output_dir + "/grammar.%d.gz"%sent_num, 'wb')
        translate_line(line.strip(), inventory, extractor, left_con, right_con, bidi_lowrank_con, tokens, out_fh)
        out_fh.close()

if __name__ == "__main__":
    main()
