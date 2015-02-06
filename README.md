cca-mt
======
`cca-mt` is a package for phrase disambiguation in machine translation (MT). 

## System Requirements

- NumPy (at least v1.6, preferably v1.9) and SciPy (at least v0.13, preferably v0.15)
- MATLAB for fast SVD/CCA computation
- [cdec](http://www.cdec-decoder.org/), for MT tuning and decoding. 

## End-to-End Pipeline Instructions

Inputs: 

1. Parallel sentence corpus (tokenized and pre-processed) and word alignments. The parallel corpus should have sentence pairs separated by ` ||| `
2. Output of Zhang-Gildea-Chiang Minimal Grammar Extractor (available [here](http://www.cs.rochester.edu/u/gildea/mt/factorize-alignment.tgz))

Steps:

1. Extract minimal phrase pairs from grammar extractor output: 

   ```
   python utils/extract_rules.py parallel_corpus alignments phrase_pair_dir phrase_pair_counts < extractor_output
   ```

   The script writes out the phrase pairs on a per-sentence basis, each sentence in a separate .gz file in the `phrase_pair_dir` location. 

   The phrase pair counts are also written, for downstream processing (when featurizing the grammars). 

   Options:
   - `-l`: lexical model; outputs translations from word alignments.  `STDIN` is ignored if enabled. 
   - `-m X`: maximum length of phrase pair.  Default is 6. 

2. Train the model:

   ```
   python train.py phrase_pair_dir parameters < parallel_corpus
   ```

   The script writes out the parameters in the location specified by the `parameters` arguments.    

   Options:
   - `-c`: concatenative model (representations for individual context words are contenated). Default is additive model. 
   - `-C X`: context parameters location; if the file does not exist it will write out the context parameters to the location, otherwise it will read them in from this location. 
   - `-f X`: restrict context features (words) to provided list.  For example, the list could contain the N most frequent words in the language and we only estimate context representations for these words (other words get mapped to `<OTHER>`). 
   - `-g X,Y`: regularizer strengths. There are two regularizers generally: one for the CCA between left and right contexts, and one for the model.  For some setups (e.g., if whitening is set to `identity`) there is only one regularizer. If only one regularizer is specified, then both regularizers will have this value.  Default is 1e-5. 
   - `-h X`: held-out percentage for mean reciprocal rank (MRR) computation. 
   - `-l X`: context length (on one side). Default is 2. 
   - `-m X`: model after context CCA is computed.  Can be one of `cca`, `glm`, or `mlp`.  Default is `cca`. 
   - `-M`: mean-center left and right context matrices. If this is enabled and `-m cca` as above, then the second CCA step will involve mean-centering. 
   - `-o X`: count of context types to consider for OOV.  Context words that occur less than or `X` times are not included in the list of features and instead their occurrences are used to estimate OOV feature representations. Default is 1. 
   - `-p`: enable position-dependent features for context.  Default is bag-of-words. 
   - `-P`: enable singleton pruning for phrase pairs.
   - `-s X`: filter stop words from context, where the stop word list is provided in `X`. 
   - `-S X`: for James-Stein shrinkage estimation.  Only valid if `-m glm` is enabled, and one cannot have both `-h X` and `-S X` enabled. In this setting, we refine the initial parameters estimated using the normal equations using the shrinkage technique described in R. Shah et al., EMNLP 2010, and `X` defines the size of the set (as a fraction) to use for this shrinkage. 
   - `-r X,Y`: ranks.  If only one rank is provided, then both ranks will have this value. Some models (`cca`, `mlp`) require two ranks, one for the context and one for the model itself.  Defaults are 100 and 50. 
   - `-v X`: use dense word vectors for features instead of sparse, one-hot feature representations.  The word vectors must be estimated elsewhere e.g., using the [word2vec](https://code.google.com/p/word2vec/) software. 
   - `-w X`: Rescaling of cross-covariance matrix. can be one of `identity`, `diag`, `ppmi`, or `full`.  Default is `full`. 

3. Use the learned parameters to score the relevant phrase pairs for given development/test sets:

   ```
   bash decode_pipeline.sh scripts_loc cdec_loc working_dir parameters parallel_dev parallel_test phrase_pair_counts baseline_lex_model config_file baseline_weights num_processes
   ```

   The script evaluates and scores `parallel_dev` and `parallel_test` (with un-normalized and normalized scores, where normalization is done with respect to a source phrase) and computes BLEU scores using 3 MIRA setups: initialization with the baseline set of weights with and without score normalization, and random weight initialization (un-normalized).  `phrase_pair_counts` and `baseline_lex_model` are used for featurizing the rules similar to the baseline, and `config_file` and `baseline_weights` are used for decoding and MIRA tuning. 
