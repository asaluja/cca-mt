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

def read_phrase_pairs(filename):
    filehandle = open(filename, 'rb')
    phrase_dict = {}
    for line in filehandle:
        elements = line.strip().split(' ||| ')
        src = elements[0]
        tgt = elements[1]
        translations = phrase_dict[src] if src in phrase_dict else []
        translations.append(tgt)
        phrase_dict[src] = translations
    return phrase_dict

'''
currently implements lexical (word-by-word) translation
'''
def translate_line(sentence, inventory, extractor, left_con, right_con, lowrank_con, tokens):
    words = sentence.split()
    for idx, word in enumerate(words):
        if word in inventory: #we have seen source word in training before and have at least one translation
            translations = inventory[word]
            representations = []
            for translation in translations:
                phrase_pair = ' ||| '.join([word, translation])
                representations.append((phrase_pair, tokens.get_representation([phrase_pair]))) #guaranteed that get_representation returns a non-NULL vector, because word is in inventory
            left_con_words, right_con_words = extractor.extract_context(words, idx, idx)
            left_con_lr = left_con.get_representation(left_con_words) #list is in order, closest is last
            right_con_lr = right_con.get_representation(right_con_words) #list is reversed, closest is first
            if left_con_lr is not None and right_con_lr is not None:
                concat_con_lr = np.concatenate((left_con_lr, right_con_lr), axis=1)
                bidi_con_lr = lowrank_con.project(concat_con_lr)
                print "Word: '%s'; Left context: '%s'; Right context: '%s'"%(word, ' '.join(left_con_words), ' '.join(reversed(right_con_words)))
                scored_pps = []            
                for phrase_pair, representation in representations:
                    score = representation.dot(bidi_con_lr.transpose()) / (np.linalg.norm(representation)*np.linalg.norm(bidi_con_lr))
                    scored_pps.append((phrase_pair, score))
                sorted_pps = sorted(scored_pps, key=lambda x: x[1], reverse=True)
                for pp, score in sorted_pps:
                    print "Translation: '%s'; Score: %.3f"%(pp.split(' ||| ')[1], score)
            else:
                left_null = left_con_lr is None
                null_context_side = "left" if left_null else "right"
                null_context = ' '.join(left_con_words) if left_null else ' '.join(reversed(right_con_words))
                print "Word: '%s'; Context on %s ('%s') is completely OOV"%(word, null_context_side, null_context)
        else:
            print "Word '%s' is OOV"%word

def main():
    args = sys.argv[1:]
    inventory = read_phrase_pairs(args[0])
    paramDict = cPickle.load(open(args[1], 'rb'))
    left_con = paramDict["left_con"]
    right_con = paramDict["right_con"]
    bidi_lowrank_con = paramDict["lowrank_con"]
    tokens = paramDict["tokens"]
    con_length = paramDict["con_length"]
    pos_depend = paramDict["pos_depend"]
    for sent_num, line in enumerate(sys.stdin):
        extractor = context_extractor(con_length, pos_depend)
        print "Sentence Number: %d"%sent_num
        translate_line(line.strip(), inventory, extractor, left_con, right_con, bidi_lowrank_con, tokens)

if __name__ == "__main__":
    main()
