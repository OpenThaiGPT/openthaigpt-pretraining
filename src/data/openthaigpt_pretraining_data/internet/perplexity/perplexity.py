import kenlm
import math
import numpy as np
import pandas as pd
import pickle
import scipy
import sentencepiece  # type: ignore
from openthaigpt_pretraining_data.core.text_normalizer import normalize
from typing import List
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class SentencesLM:
    """Returns the score of each individual paragraph."""

    def __init__(self):
        lm_config = kenlm.Config()
        lm_config.load_method = 2

        lm_model_filename = "../../openthaigpt_pretraining_data/internet/perplexity/th.arpa.bin"
        self.lm = kenlm.Model(str(lm_model_filename), lm_config)
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load("../../openthaigpt_pretraining_data/internet/perplexity/th.sp.model")

    def pp(self, log_score: float, length: int) -> float:
        """Compute perplexity score"""
        power = min(30, -log_score / length)

        return 10.0**power

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


classifier_filename = (
    "../../openthaigpt_pretraining_data/internet/perplexity/decision_tree.sav"
)
classifier = pickle.load(open(classifier_filename, "rb"))

lm = SentencesLM()


def classify_spam(text: str):
    """Classify if text is spam using perplexity and decision tree as thresholder
    Input : text -> a text to classify.
    Output : prediction -> Prediction whether text is spam.
                    1 Represents spam and 0 represent non-spam.
    Output : log_pp_score -> log of perplexity score.
    """

    pp_score = lm.do(text.split("\n"))

    log_pp_score = math.log(pp_score)

    prediction = classifier.predict(pd.DataFrame({"log_score": [log_pp_score]}))

    return prediction, log_pp_score


def sample_text_back(
    probs: np.ndarray,
    percentage: float = 0.1,
) -> List[int]:
    """Sample some spam text back in the dataset
    using log score distribution of language model

    Input : spam_data_points -> data points which its text classified as spam
                            by perplexity.
    Input : probs -> prob of log perplexity.
    Input : percentage -> percent of data to sample back.

    Output : sampled_data -> The sampled back data.
    """

    n = len(probs)
    if n <= 1:
        return []

    norm_probs = scipy.special.softmax(1 - probs)
    np.random.seed(0)

    selected_idx = np.random.choice(
        n, p=norm_probs, size=int(percentage * n), replace=False
    )

    return list(selected_idx)
