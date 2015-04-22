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
numProc=$9

setup_dev=feat-dev-rep-disc
sgm_dev=dev.rep.disc.sgm
setup_devtest=feat-devtest-rep-disc
sgm_devtest=devtest.rep.disc.sgm
python_command="python ${scripts}/decode.py -r -d /usr0/home/avneesh/cca-translate/data/EMNLP/zh-en/fbis/cca/context+tokens.sparse"
#can add various arguments to command above

#dev
${cdec}/corpus/cut-corpus.pl 1 < $devSrcTgt > ${working}/dev.src
mkdir ${working}/score-dev/

${python_command} $params ${working}/score-dev/ $numProc < ${working}/dev.src 
mkdir ${working}/${setup_dev}/
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-dev/ ${working}/${setup_dev}/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/${setup_dev}/ < $devSrcTgt > ${working}/${sgm_dev}

rm -rf ${working}/score-dev/
rm ${working}/dev.src

#devtest
${cdec}/corpus/cut-corpus.pl 1 < $testSrcTgt > ${working}/devtest.src
mkdir ${working}/score-devtest/

${python_command} $params ${working}/score-devtest/ $numProc < ${working}/devtest.src 
mkdir ${working}/${setup_devtest}
python ${scripts}/utils/featurize_rules.py -a -s ${working}/score-devtest/ ${working}/${setup_devtest}/ $counts $lexModel $numProc 
python ${scripts}/utils/corpus2sgm.py ${working}/${setup_devtest}/ < $testSrcTgt > ${working}/${sgm_devtest}

rm -rf ${working}/score-devtest/
rm ${working}/devtest.src


