#!/usr/bin/python -tt

'''
File: add_target_context.py
Date: March 24, 2015
Description: adds target context to phrase pairs
Inputs:
arg1: per-sentence minimal phrase pair grammars
arg2: word alignments
arg3: context length (on each side), integer
arg4: output directory
to do: change output from sentence-level to writing out to one big file
'''

import sys, commands, string, gzip, collections

def main():
    pp_dir_loc = sys.argv[1]
    alignments_list = open(sys.argv[2], 'r').read().splitlines()
    con_length = int(sys.argv[3])
    pp_dir_out_loc = sys.argv[4]
    for count, line in enumerate(sys.stdin):
        alignments = collections.defaultdict(list)
        align_pairs = alignments_list[count].strip().split()
        target_list = line.strip().split(' ||| ')[1].split()
        for align_pair in align_pairs:
            k, v = tuple([int(idx) for idx in align_pair.split('-')])
            alignments[k].append(v)
        pp_fh = gzip.open(pp_dir_loc + "/grammar.%d.gz"%count, 'rb')
        pp_out_fh = gzip.open(pp_dir_out_loc + "/grammar.%d.gz"%count, 'wb')
        for rule in pp_fh:
            elements = rule.strip().split(' ||| ')            
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
                if left_con_idx < 0:
                    #left_context.append("<s>")
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
            line_out = "%s %s %s"%(left_con_str, elements[1], right_con_str)
            pp_out_fh.write("%s\n"%line_out.lstrip().rstrip())
            #pp_out_fh.write("%s ||| %s ||| %s ||| %s ||| %s\n"%(elements[0], elements[1], elements[2], ' '.join(left_context), ' '.join(right_context)))
        pp_fh.close()
        pp_out_fh.close()

if __name__ == "__main__":
    main()
