#!/usr/bin/env python

'''
File: source_channel.py
Date: April 22, 2015
Description: class implementation of source_channel class to compute
MRR using LM and other information
'''

import sys, commands, string, math

class SourceChannel:
    def __init__(self, lms, tim, counts, rev_counts):
        self.lm_scores = lms #lms is a list of dicts, indexed by example ID; only the heldout example IDs are passed in
        self.type_id_map = tim
        self.counts = counts
        self.reverse_counts = rev_counts

    def score_all(self, index, src_phrase): 
        scored_lm = []
        scored_fwd = []
        scored_lm_fwd = []
        scored_rev = []
        scored_lm_rev = []
        lm_dict = self.lm_scores[index]
        tgt_phrase_counts = self.counts[src_phrase]
        src_normalizer = float(sum(tgt_phrase_counts.values())) #denominator in P(e|f)
        for tgt_phrase in lm_dict: 
            phrase_pair = src_phrase + " ||| " + tgt_phrase
            lm_score = -lm_dict[tgt_phrase] if phrase_pair in self.type_id_map else None
            if lm_score is not None:
                scored_lm.append((phrase_pair, lm_score, ""))
                e_given_f = math.log10(tgt_phrase_counts[tgt_phrase] / src_normalizer)
                scored_fwd.append((phrase_pair, e_given_f, ""))
                scored_lm_fwd.append((phrase_pair, lm_score+e_given_f, ""))
                src_phrase_counts = self.reverse_counts[tgt_phrase] 
                tgt_normalizer = float(sum(src_phrase_counts.values()))
                f_given_e = math.log10(src_phrase_counts[src_phrase] / tgt_normalizer)
                scored_rev.append((phrase_pair, f_given_e, ""))
                scored_lm_rev.append((phrase_pair, lm_score+f_given_e, ""))                
            else:
                result = (phrase_pair, None, "")
                scored_lm.append(result)
                scored_fwd.append(result)
                scored_lm_fwd.append(result)
                scored_rev.append(result)
                scored_lm_rev.append(result)
        return scored_lm, scored_fwd, scored_lm_fwd, scored_rev, scored_lm_rev        
