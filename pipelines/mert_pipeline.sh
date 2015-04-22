#!/usr/bin/bash

cdec=$1
working=$2
config=$3
weights_init=$4
devSrcTgt=$5
testSrcTgt=$6
numProc=$7

${cdec}/training/dpmert/dpmert.pl --config $config --devset $devSrcTgt --output-dir ${working}/mert/ --weights $weights_init --jobs $numProc &> ${working}/mert.log
${cdec}/training/utils/decode-and-evaluate.pl --jobs $numProc --config $config --input $devSrcTgt --weights ${working}/mert/weights.final &> ${working}/mert/dev.bleu
${cdec}/training/utils/decode-and-evaluate.pl --jobs $numProc --config $config --input $testSrcTgt --weights ${working}/mert/weights.final &> ${working}/mert/devtest.bleu
rm -rf eval.*
