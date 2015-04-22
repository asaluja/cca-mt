#!/usr/bin/python -tt

'''
File: analyze_translations.py
Date: April 10, 2015
Arguments:
STDIN: per-sentence grammar (in unzipped format)
arg: span of desired source phrase
output: ranked translations by cca_score
'''

import sys, commands, string, re

span_dict = {}
expr = re.compile(r'\[([^]]*)\]')
for line in sys.stdin:
    elements = line.strip().split(' ||| ')
    span_str = expr.match(elements[0]).group(1)
    if span_str != "S":
        NTcat, left, right = span_str.split('_')    
        features = elements[3].split()
        score = float(features[1].split('=')[1]) if features[0] == "cca_on=1" else None
        key = (int(left), int(right))
        translations = span_dict[key] if key in span_dict else []
        value = (elements[1], elements[2], score)
        translations.append(value)
        span_dict[key] = translations

span_str = sys.argv[1]
left = int(span_str.split('-')[0])
right = int(span_str.split('-')[1])
desired_key = (left, right)
if desired_key in span_dict:
    translations = span_dict[desired_key]
    sorted_translations = sorted(translations, key=lambda x: x[2], reverse=True)
    for src_phrase, tgt_phrase, score in sorted_translations:
        print "%s ||| %s ||| %s"%(src_phrase, tgt_phrase, score)
else:
    print "Desired span not in dictionary!"

    
    
        
    
