# Perplexity preprocessing code

This folder contains codes to preprocess data using the perplexity score computed from a kenlm 5-gram language model trained on Thai wikipedia data.

This idea are originally from [Perplexed by Quality](https://arxiv.org/pdf/2212.10440).pdf"

The main code is in `perplexity.py`

The `notebook` folder contains the experiment, observation and EDA notebooks for perplexity method.

## How does the code work ?

1. The perplexity score of text will be computed from the language model and taken log.

2. The log score will be used as a feature of DecisionTree classifier to predict if the text is inappropriate. This classifier is train on the created dataset from sampled OSCAR2023.

3. After classified the amount of text, Inappropriate text from step 2 will be sample back to our training set to teach inappropriate terms to our LLM and retain the amount of data in step 2. 

    3.1 The distribution of logs-perplexity score will be formed.
    3.2 Compute the PDF (Probability Density Function) of the each log score 
    3.3 softmax(1-PDF) will be used as probability list for np.choice to sample text back. 

## Running

The code is imported in `src/data/scripts/internet` and you can use it together with mc4 and cc100 regex code there. 

This code is not meant to be run directly. If you want to run with your custom logic please create folder in `src/data/scripts` and import the function you want.

## Note



