import kenlm
import math
import numpy as np
import pickle
import scipy
import sentencepiece  # type: ignore
from text_normalizer import normalize
from typing import List


class SentencesLM:
    """Returns the score of each individual paragraph."""

    def __init__(self):
        lm_config = kenlm.Config()
        lm_config.load_method = 2

        lm_model_filename = "th.arpa.bin"
        self.lm = kenlm.Model(str(lm_model_filename), lm_config)
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load("th.sp.model")

    def pp(self, log_score: float, length: int) -> float:
        """Compute perplexity score"""
        return 10.0 ** (-log_score / length)

    def do(self, document: List[str]) -> float:  # type: ignore
        """Compute perplexity for each line of document"""
        total_pp = 0
        total_length = 0
        for line in document:
            line = normalize(line, accent=False)
            tokenized_line = " ".join(self.sp.encode_as_pieces(line))
            log_score = self.lm.score(tokenized_line)
            length = len(line.split()) + 1

            total_length += length
            total_pp += log_score
        return round(self.pp(total_pp, total_length), 1)


classifier_filename = "decision_tree.sav"
classifier = pickle.load(open(classifier_filename, "rb"))

lm = SentencesLM()


def classify_spam(text: str):
    """Classify if text is spam using perplexity and decision tree as thresholder
    Input : text -> a text to classify.
    Output : prediction -> Prediction whether text is spam.
                    1 Represents spam and 0 represent non-spam.
    Output : logg_pp_score -> log of perplexity score.
    """

    pp_score = lm.do(text.split("\n"))

    log_pp_score = math.log(pp_score)

    prediction = classifier.predict(np.array([log_pp_score]).reshape(1, 1))

    return prediction, log_pp_score


def sample_score(log_scores, percentage=0.1) -> np.ndarray:
    """Sample score to use in function sample_text_back

    Input : log_scores -> log of perplexity scores of texts.
    Input : percentage -> percent of data to sample back.
    Input : replace -> If True the sampled data can be duplicated.

    Output : sampled_texts -> The sampled texts.
    """
    np.random.seed(0)

    n = len(log_scores)

    lower_bound, upper_bound = min(log_scores), max(log_scores)

    mean, std = np.mean(log_scores), np.std(log_scores)

    sampled_scores = scipy.stats.truncnorm.rvs(
        (lower_bound - mean) / std,
        (upper_bound - mean) / std,
        loc=mean,
        scale=std,
        size=int(percentage * n),
    )

    return sampled_scores


def sample_text_back(texts, log_scores, percentage=0.1, replace=True) -> List[str]:
    """Sample some spam text back in the dataset
    using log score distribution of language model

    Input : texts -> texts classified as spam by perplexity.
    Input : log_scores -> log of perplexity scores of texts.
    Input : percentage -> percent of data to sample back.
    Input : replace -> If True the sampled data can be duplicated.

    Output : sampled_texts -> The sampled texts.
    """

    sampled_scores = sample_score(log_scores, percentage)

    sampled_texts = []

    selected_idx = set()

    for samp_score in sampled_scores:
        min_diff, min_idx = float("inf"), -1

        for idx, s in enumerate(log_scores):
            if idx in selected_idx:
                continue

            diff = (samp_score - s) ** 2
            if diff < min_diff:
                min_diff = diff
                min_idx = idx

        sampled_texts.append(texts[min_idx])

        if not replace:
            selected_idx.add(min_idx)

    return sampled_texts
