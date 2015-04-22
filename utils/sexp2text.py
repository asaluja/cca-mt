#!/usr/bin/python -tt

'''
Description: quick script wrapping around NLTK that reads in
s-expressions, strips out the hierarchical information and
prints out the leaves (words). 
'''

import sys, commands, string
import nltk

for line in sys.stdin:
    t = nltk.Tree(line.strip())
    print ' '.join(t.leaves())
