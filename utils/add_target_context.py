#!/usr/bin/python -tt

'''
File: add_target_context.py
Date: March 24, 2015
Description: adds target context to phrase pairs
Inputs:
arg1: per-sentence minimal phrase pair grammars
arg2: word alignments
arg3: context length (on each side), integer
arg4: rule counts dictionary and filter cut-off (for P(e|f) pruning)
arg5: output filename
STDIN: parallel sentence corpus
'''

import sys, commands, string, gzip, collections, cPickle

def filter_counts_dict(counts_dict, cutoff):
    excluded_PPs = set()
    new_counts_dict = {}
    for src_phrase in counts_dict:
        sorted_translations = sorted(counts_dict[src_phrase], key=counts_dict[src_phrase].get, reverse=True)
        new_counts_dict[src_phrase] = sorted_translations[:cutoff] if len(sorted_translations) > cutoff else sorted_translations
    return new_counts_dict

def main():
    pp_dir_loc = sys.argv[1]
    alignments_list = open(sys.argv[2], 'r').read().splitlines()
    con_length = int(sys.argv[3])
    count_filename, filter_cutoff = sys.argv[4].split(',')
    counts_fh = open(count_filename, 'rb')
    counts_dict = cPickle.load(counts_fh)    
    counts_fh.close()
    counts_dict = filter_counts_dict(counts_dict, int(filter_cutoff)) #filtered and sorted
    pp_out_loc = sys.argv[5]
    pp_out_fh = open(pp_out_loc, 'wb')
    for count, line in enumerate(sys.stdin):
        alignments = collections.defaultdict(list)
        align_pairs = alignments_list[count].strip().split()
        target_list = line.strip().split(' ||| ')[1].split()
        for align_pair in align_pairs:
            k, v = tuple([int(idx) for idx in align_pair.split('-')])
            alignments[k].append(v)
        pp_fh = gzip.open(pp_dir_loc + "/grammar.%d.gz"%count, 'rb')
        for rule in pp_fh:
            elements = rule.strip().split(' ||| ')
            tgt_phrase = elements[1]
            left_idx = int(elements[2].split('-')[0])
            right_idx = int(elements[2].split('-')[1])
            assert left_idx <= right_idx
            indices = []            
            for idx in range(left_idx, right_idx+1):
                if idx in alignments:
                    indices += alignments[idx]
            if len(indices) == 0:
                print count, target_list
                print alignments
                sys.exit()
            min_idx = min(indices)
            max_idx = max(indices)
            left_con_idx = min_idx - con_length
            left_context = []
            while left_con_idx < min_idx:
                if left_con_idx < 0: #choose not to append <s> to left context since this screws up LM scoring
                    left_con_idx = 0
                else:
                    left_context.append(target_list[left_con_idx])
                    left_con_idx += 1
            right_con_idx = max_idx + 1
            right_context = []
            while right_con_idx < max_idx + con_length + 1:
                if right_con_idx >= len(target_list):
                    right_context.append("</s>")
                    break
                else:
                    right_context.append(target_list[right_con_idx])
                    right_con_idx += 1
            left_con_str = ' '.join(left_context) if len(left_context) > 0 else ""
            right_con_str = ' '.join(right_context) if len(right_context) > 0 else ""
            src_phrase = elements[0]
            tgt_candidates = list(counts_dict[src_phrase]) #need to make a copy of the list since we modify it downstream
            if tgt_phrase not in tgt_candidates: #will not be in tgt_candidates if filtered out
                print "Phrase '%s' has been filtered out; adding back in"%tgt_phrase
                assert len(tgt_candidates) == int(filter_cutoff)                
                tgt_candidates = tgt_candidates[:-1]
            else: #just place right answer in front
                tgt_candidates.remove(tgt_phrase) #temporarily remove correct answer
            tgt_candidates = [tgt_phrase] + tgt_candidates
            for tgt_candidate in tgt_candidates: #at this stage, we guarantee that the order of the scores is such that the first score is the correct answer and the remaining (up to 19 more) scores are the negative examples
                line_out = "%s %s %s"%(left_con_str, tgt_candidate, right_con_str)
                print >> pp_out_fh, "%s ||| %s ||| %s"%(src_phrase, tgt_candidate, line_out.lstrip().rstrip())
            print >> pp_out_fh, ""
        pp_fh.close()
    pp_out_fh.close()

if __name__ == "__main__":
    main()
