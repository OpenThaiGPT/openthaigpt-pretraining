# Perplexity preprocessing code

This folder contains codes to preprocess data using the perplexity score computed from a kenlm 5-gram language model trained on Thai wikipedia data.

The main code is in `perplexity.py`

The `notebook` folder contains the experiments, observations and EDA notebooks for perplexity method.

## How does the code work ?

1. The perplexity score of texts will be computed from the language model and taken log.

2. The log-perplexity score will be used as a feature of DecisionTree classifier to predict if the text is garbage. 

3. After classified the amount of text, sampled set of garbage text **S** from step 2 will be sample back and add to training set to teach inappropriate terms to our LLM. Here are the detailed steps.

    - The normal distribution of log-perplexity score will be formed.
    - Compute the PDF (Probability Density Function) of the each log score.
    - Softmax(1-PDF) will be used as probability list for `np.choice` to sample text back. 

## Running

The code is imported in `src/data/scripts/internet` and you can use it together with mc4 and cc100 regex code there. 

This code is not meant to be run directly. If you want to run with your custom logic please create folder in `src/data/scripts` and import the function you want.

## Note

- The idea of using perplexity score are originally from [Perplexed by Quality](https://arxiv.org/pdf/2212.10440.pdf).
- **_However, we decided to try another method on the score since the perplexity score and thresholding is not enough to classify bad data acoording to the observation._**
- DecisionTree classifier was train on the sampled OSCAR2023.



