#!/usr/bin/bash

#score computation, decoration of rules, MIRA for dev and devtest
scripts=$1
cdec=$2
working=$3
params=$4
devSrcTgt=$5
testSrcTgt=$6
counts=$7
lexModel=$8
config=$9
weights_init=${10}
numProc=${11}

#dev
${cdec}/corpus/cut-corpus.pl 1 < $devSrcTgt > ${working}/dev.src
mkdir ${working}/score-dev/

python ${scripts}/decode.py $params ${working}/score-dev/ $numProc < ${working}/dev.src 
mkdir ${working}/feat-dev/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-dev/ ${working}/feat-dev/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-dev/ < $devSrcTgt > ${working}/dev.sgm

python ${scripts}/decode.py -n $params ${working}/score-dev/ $numProc < ${working}/dev.src 
mkdir ${working}/feat-dev-norm/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-dev/ ${working}/feat-dev-norm/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-dev-norm/ < $devSrcTgt > ${working}/dev_norm.sgm

rm -rf ${working}/score-dev/
rm ${working}/dev.src


#devtest
${cdec}/corpus/cut-corpus.pl 1 < $testSrcTgt > ${working}/devtest.src
mkdir ${working}/score-devtest/

python ${scripts}/decode.py $params ${working}/score-devtest/ $numProc < ${working}/devtest.src 
mkdir ${working}/feat-devtest/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-devtest/ ${working}/feat-devtest/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-devtest/ < $testSrcTgt > ${working}/devtest.sgm

python ${scripts}/decode.py -n $params ${working}/score-devtest/ $numProc < ${working}/devtest.src 
mkdir ${working}/feat-devtest-norm/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-devtest/ ${working}/feat-devtest-norm/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-devtest-norm/ < $testSrcTgt > ${working}/devtest_norm.sgm

rm -rf ${working}/score-devtest/
rm ${working}/devtest.src

#MIRA tuning
${cdec}/training/mira/mira.py -d ${working}/dev.sgm -t ${working}/devtest.sgm -c $config -j $numProc -o ${working}/mira --max-iterations 10 -w $weights_init
${cdec}/training/mira/mira.py -d ${working}/dev.sgm -t ${working}/devtest.sgm -c $config -j $numProc -o ${working}/mira.rand-init --max-iterations 10
${cdec}/training/mira/mira.py -d ${working}/dev_norm.sgm -t ${working}/devtest_norm.sgm -c $config -j $numProc -o ${working}/mira.score-norm --max-iterations 10 -w $weights_init


