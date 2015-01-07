#!/usr/bin/python -tt

'''
File: featurize_rules.py (originally: decorate_minrules_with_features.py)
Date: October 13, 2013
Author: Avneesh Saluja (avneesh@cs.cmu.edu)
Description: decorates grammars either from the grammar extraction stage (post-processing
of output from ZGC extractor) or per-sentence grammars with rule marginals, with additional
features that are seen in conventional MT systems (joint phrase count, source phrase count, etc.)
This script takes as arguments the following:
arg 1: location of files with rules generated through the post-processed minimal
rule output (of of H. Zhang, D. Gildea, and D. Chiang, NAACL 2008)
arg 2: location of files with full set of rules extracted from suffix array structure
(these rules are already featurized) (update - May 19, 2014: this is not used anymore; update - August 26, 2014: this has been removed)
arg 3: location of output files 
Usage: python decorate_minrules_with_features.py minRules-dir fullRules-dir output-dir
Update (Jan 4, 2014): included per-sentence grammar writing ability
Update (Jan 6, 2014): included filtering capability based on MLE for non per-sentence grammar writing
Update (Jan 7, 2014): made a multiprocess version of this code
Update (May 16, 2014): combined this script with inside-outside/featurize_rules.py, and renamed it
featurize_rules.py
Usage: python featurize_rules.py minrules-dir/spectral-marginal-dir hiero-dir/ output-dir/ numPartitions 
There are also 3 optional flags:
-f N: filter rules by P(e|f).  The argument N is how many rules to keep for a given source RHS. -f and -m
cannot be on together. 
-m: meaning the input grammar is the output of intersect_scfg.py (contains marginals), and not the output
of tree_to_rule.py (tool to convert ZGC minimal grammar extractor output to input of feature_extraction.py
or a format that cdec can read in).  
-s: write out a per sentence grammar.  This is done by default if -m is one.  -f and -s cannot be on 
together, because we need to aggregate information. 
Update (May 19, 2014): added functionality to read in counts and compute features. 
Update (August 26, 2014): removed unnecessary arguments (previously was reading in Hiero grammar as well,
since features were being extracted from these grammars previously but now we extract these features directly 
from the rule counts that are read in. 
To do: update all of above with new thing
'''

import sys, commands, string, gzip, os, getopt, re, cPickle, math
import multiprocessing as mp
import cdec.configobj
import cdec.sa

countDict = None
MAXSCORE=99

'''
Used to maintain the top 'limit' rules when sorted by P(e|f)
Is not used if we are featurizing decorated per-sentence grammars
(which already have the rule marginal feature)
'''
def filterRules(countDict, limit):
    srcTgtDict = {}
    srcTgtDict = countDict
    for srcKey in srcTgtDict:
        numTgtRules = len(srcTgtDict[srcKey])
        if numTgtRules > limit: #then we need to filter rules
            sorted_tgtRules = sorted(srcTgtDict[srcKey], key=srcTgtDict[srcKey].get, reverse=True)
            rules_to_filter = sorted_tgtRules[limit:]
            for rule in rules_to_filter:
                srcTgtDict[srcKey].pop(rule)
            if len(rules_to_filter) > 0:
                sys.stderr.write("Source RHS: %s; out of %d rules, filtered %d\n"%(srcKey, numTgtRules, len(rules_to_filter)))
    return srcTgtDict

'''
Core feature computation function. 
'''
def computeFeatures(ruleToPrint, addOne):
    elements = ruleToPrint.split(' ||| ')
    outputStr = ""
    srcKey = elements[0]
    tgtKey = elements[1]
    if srcKey in countDict and tgtKey in countDict[srcKey]:
        normalizer = sum([countDict[srcKey][key] for key in countDict[srcKey]])
        jointCount = countDict[srcKey][tgtKey]
        EgivenF = -math.log10(jointCount / float(normalizer))
        outputStr += " EgivenF=-0.0" if EgivenF == 0 else " EgivenF=%.11f"%EgivenF
        CountF = math.log10(normalizer+len(countDict[srcKey])) if addOne else math.log10(normalizer)
        outputStr += " SampleCountF=%.11f"%CountF
        CountEF = math.log10(jointCount+1) if addOne else math.log10(jointCount)
        outputStr += " CountEF=%.11f"%CountEF
        IsSingletonF = 1 if normalizer == 1 else 0
        IsSingletonFE = 1 if jointCount == 1 else 0
        outputStr += " IsSingletonF=%d"%IsSingletonF
        outputStr += " IsSingletonFE=%d"%IsSingletonFE
    return outputStr

def maxLexEgivenF(fwords, ewords, ttable):
    local_f = fwords + ['NULL']
    maxOffScore = 0.0
    for e in ewords:
        maxScore = max(ttable.get_score(f, e, 0) for f in local_f)
        maxOffScore += -math.log10(maxScore) if maxScore > 0 else MAXSCORE
    return maxOffScore

def maxLexFgivenE(fwords, ewords, ttable):
    local_e = ewords + ['NULL'] 
    maxOffScore = 0.0
    for f in fwords:
        maxScore = max(ttable.get_score(f, e, 1) for e in local_e)
        maxOffScore += -math.log10(maxScore) if maxScore > 0 else MAXSCORE
    return maxOffScore

def computeLexicalScores(model_loc, rules_list):
    tt = cdec.sa.BiLex(from_binary=model_loc)    
    new_rules = []
    for rule in rules_list:
        new_rule = rule
        elements = rule.split(' ||| ')
        srcPhrase = elements[0]
        tgtPhrase = elements[1]
        srcWords = srcPhrase.split()
        tgtWords = tgtPhrase.split()
        if len(srcWords) > 0: 
            f_given_e = maxLexFgivenE(srcWords, tgtWords, tt)     
            if f_given_e < MAXSCORE:
                new_rule += " MaxLexFgivenE=%.11f"%f_given_e
        if len(tgtWords) > 0:
            e_given_f = maxLexEgivenF(srcWords, tgtWords, tt)
            if e_given_f < MAXSCORE:
                new_rule += " MaxLexEgivenF=%.11f"%e_given_f
        new_rules.append(new_rule)
    return new_rules            

def decorateSentenceGrammar(minRule_file, out_file, lex_model, optDict):
    perSent = "perSentence" in optDict
    addOne = "addOne" in optDict
    numRulesTotal = 0
    if os.path.isfile(minRule_file):
        rules_output = []
        minrule_fh = gzip.open(minRule_file, 'rb')
        for rule in minrule_fh:                                
            numRulesTotal += 1
            elements = rule.strip().split(' ||| ')
            key = ' ||| '.join(elements[:2])
            ruleToPrint = rule.strip()
            if not perSent: #writing out full grammar, so strip alignment info
                ruleToPrint = key + " ||| "
            if elements[0] == "<unk>":
                ruleToPrint = "%s ||| %s ||| %s PassThrough=1"%(elements[1], elements[1], elements[2]) if len(elements) > 2 else "%s ||| %s ||| PassThrough=1"%(elements[1], elements[1])
            ruleToPrint += computeFeatures(key, addOne)
            rules_output.append(ruleToPrint)
        minrule_fh.close()
        if perSent:
            new_rules = computeLexicalScores(lex_model, rules_output)
            out_fh = gzip.open(out_file, 'w')
            for ruleToPrint in new_rules:
                out_fh.write("%s\n"%ruleToPrint)
        else:
            seen_rules.extend(rules_output) #global write
        print "Grammar %s featurization complete: %d rules"%(minRule_file, numRulesTotal)
    if perSent: #add the NT only rules
        out_fh.write("[X] ||| [X,1] [X,2] ||| [1] [2] ||| Glue=1\n")
        out_fh.write("[X] ||| [X,1] [X,2] ||| [2] [1] ||| Glue=1 Inverse=1\n")            
        out_fh.write("[S] ||| [X,1] ||| [1] ||| 0\n") #no features defined on the top-level rule, just for parsing completion purposes
        out_fh.close()

def init(sr):
    global seen_rules
    seen_rules = sr

def main():
    global countDict    
    optDict = {}
    (opts, args) = getopt.getopt(sys.argv[1:], 'af:s')
    for opt in opts:
        if opt[0] == '-a': #addOne
            optDict["addOne"] = 1
        elif opt[0] == '-f':
            optDict["filterRules"] = int(opt[1])
        elif opt[0] == '-s': #write out per sentence
            optDict["perSentence"] = 1
    if "filterRules" in optDict and "perSentence" in optDict:
        sys.stderr.write("Error: -f and -s cannot be on at the same time\n")
        sys.exit()
    minRule_grammars_loc = args[0]
    outFile_loc = args[1]
    countDict = cPickle.load(open(args[2], 'rb')) #load counts for feature computation
    lex_model = args[3]
    numProcesses = int(args[4])
    
    minRule_grammars = os.listdir(minRule_grammars_loc)
    seen_rules = None
    pool = None
    if "perSentence" not in optDict:
        seen_rules = mp.Manager().list()
        pool = mp.Pool(processes=numProcesses, initializer=init, initargs=(seen_rules,))
    else:
        pool = mp.Pool(numProcesses)
    for minRule_file in minRule_grammars:
        if numProcesses > 1:
            pool.apply_async(decorateSentenceGrammar, (minRule_grammars_loc + minRule_file, outFile_loc + minRule_file, lex_model, optDict))    
        else:
            decorateSentenceGrammar(minRule_grammars_loc + minRule_file, outFile_loc + minRule_file, lex_model, optDict)
    pool.close()
    pool.join()

    if "perSentence" not in optDict:                                 
        print "number of rules seen: %d"%len(seen_rules)
        output_fh = gzip.open(outFile_loc, 'wb')
        seen_rules_uniq = list(set(seen_rules))
        new_rules = computeLexicalScores(lex_model, seen_rules_uniq)
        if "filterRules" in optDict:
            filteredDict = filterRules(countDict, optDict["filterRules"])            
            for rule in new_rules:
                elements = rule.split(' ||| ')
                srcKey = elements[0]
                tgtKey = elements[1]
                if tgtKey in filteredDict[srcKey]: #i.e., we haven't pruned it away
                    output_fh.write("%s\n"%(rule))
        else:
            for rule in new_rules:
                output_fh.write("%s\n"%(rule))
        output_fh.write("[X] ||| [X,1] [X,2] ||| [1] [2] ||| Glue=1\n")
        output_fh.write("[X] ||| [X,1] [X,2] ||| [2] [1] ||| Glue=1 Inverse=1\n")        
        output_fh.write("[S] ||| [X,1] ||| [1] ||| 0\n") #no features defined on the top-level rule, just for parsing completion purposes
        output_fh.close()

if __name__ == "__main__":
    main()
