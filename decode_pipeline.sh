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

python ${scripts}/decode.py -n exp $params ${working}/score-dev/ $numProc < ${working}/dev.src 
mkdir ${working}/feat-dev-expnorm/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-dev/ ${working}/feat-dev-expnorm/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-dev-expnorm/ < $devSrcTgt > ${working}/dev_expnorm.sgm

python ${scripts}/decode.py -n range $params ${working}/score-dev/ $numProc < ${working}/dev.src 
mkdir ${working}/feat-dev-rangenorm/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-dev/ ${working}/feat-dev-rangenorm/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-dev-rangenorm/ < $devSrcTgt > ${working}/dev_rangenorm.sgm

rm -rf ${working}/score-dev/
rm ${working}/dev.src

#devtest
${cdec}/corpus/cut-corpus.pl 1 < $testSrcTgt > ${working}/devtest.src
mkdir ${working}/score-devtest/

python ${scripts}/decode.py $params ${working}/score-devtest/ $numProc < ${working}/devtest.src 
mkdir ${working}/feat-devtest/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-devtest/ ${working}/feat-devtest/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-devtest/ < $testSrcTgt > ${working}/devtest.sgm

python ${scripts}/decode.py -n exp $params ${working}/score-devtest/ $numProc < ${working}/devtest.src 
mkdir ${working}/feat-devtest-expnorm/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-devtest/ ${working}/feat-devtest-expnorm/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-devtest-expnorm/ < $testSrcTgt > ${working}/devtest_expnorm.sgm

python ${scripts}/decode.py -n range $params ${working}/score-devtest/ $numProc < ${working}/devtest.src 
mkdir ${working}/feat-devtest-rangenorm/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-devtest/ ${working}/feat-devtest-rangenorm/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/feat-devtest-rangenorm/ < $testSrcTgt > ${working}/devtest_rangenorm.sgm

rm -rf ${working}/score-devtest/
rm ${working}/devtest.src

#MIRA tuning
${cdec}/training/mira/mira.py -d ${working}/dev.sgm -t ${working}/devtest.sgm -c $config -j $numProc -o ${working}/mira --max-iterations 10 -w $weights_init
${cdec}/training/mira/mira.py -d ${working}/dev.sgm -t ${working}/devtest.sgm -c $config -j $numProc -o ${working}/mira.rand-init --max-iterations 10
${cdec}/training/mira/mira.py -d ${working}/dev_expnorm.sgm -t ${working}/devtest_expnorm.sgm -c $config -j $numProc -o ${working}/mira.exp-norm --max-iterations 10 -w $weights_init
${cdec}/training/mira/mira.py -d ${working}/dev_rangenorm.sgm -t ${working}/devtest_rangenorm.sgm -c $config -j $numProc -o ${working}/mira.range-norm --max-iterations 10 -w $weights_init


