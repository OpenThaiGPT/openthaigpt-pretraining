import kenlm
import math
import numpy as np
import pandas as pd
import pickle
import scipy
import sentencepiece  # type: ignore
from openthaigpt_pretraining_data.core.text_normalizer import normalize
from typing import List, Dict


class SentencesLM:
    """Returns the score of each individual paragraph."""

    def __init__(self):
        lm_config = kenlm.Config()
        lm_config.load_method = 2

        lm_model_filename = "openthaigpt_pretraining_data/core/th.arpa.bin"
        self.lm = kenlm.Model(str(lm_model_filename), lm_config)
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load("openthaigpt_pretraining_data/core/th.sp.model")

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


classifier_filename = "openthaigpt_pretraining_data/core/decision_tree.sav"
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

    prediction = classifier.predict(pd.DataFrame({"log_score": [log_pp_score]}))

    return prediction, log_pp_score


def sample_score(log_scores: List[float], percentage: float = 0.1) -> np.ndarray:
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
    print(log_scores)
    sampled_scores = scipy.stats.truncnorm.rvs(
        (lower_bound - mean) / std,
        (upper_bound - mean) / std,
        loc=mean,
        scale=std,
        size=int(percentage * n),
    )

    return sampled_scores


def sample_text_back(
    spam_data_points: List[dict[str, str]],
    log_scores: List[float],
    percentage: float = 0.1,
    replace: bool = True,
) -> List[dict]:
    """Sample some spam text back in the dataset
    using log score distribution of language model

    Input : spam_data_points -> data points which its text classified as spam
                            by perplexity.
    Input : log_scores -> log of perplexity scores of texts.
    Input : percentage -> percent of data to sample back.
    Input : replace -> If True the sampled data can be duplicated.

    Output : sampled_texts -> The sampled texts.
    """
    if len(spam_data_points) <= 1:
        return []
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

        sampled_texts.append(spam_data_points[min_idx])

        if not replace:
            selected_idx.add(min_idx)

    return sampled_texts


def remove_spam_from_dataset(
    dataset: List[Dict[str, str]],
    percentage: float = 0.1,
    replace: bool = False,
) -> List[Dict[str, str]]:
    """Remove spam which being classified as spam by the classifier
       and also sample percentage percent of the spam back in the dataset
    Input : dataset -> dataset of text contains its text in 'text' field.
    Input : percentage -> percent of data to sample back.
    Input : replace -> If True the sampled data can be duplicated.

    Output : sampled_texts -> The sampled texts.
    """
    spam_data_points = []
    spam_log_scores = []

    non_spam_data_points = []

    for i, data_point in enumerate(dataset):
        prediction, log_score = classify_spam(data_point["text"])

        if prediction == 1:
            spam_data_points.append(data_point)
            spam_log_scores.append(log_score)
        else:
            non_spam_data_points.append(data_point)

    sampled_spam_texts = sample_text_back(
        spam_data_points, spam_log_scores, percentage=percentage, replace=replace
    )

    return non_spam_data_points + sampled_spam_texts
