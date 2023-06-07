from openthaigpt_pretraining_data.mc4.preprocess import (
    clean_text as clean_mc4_text,
)
from openthaigpt_pretraining_data.oscar.preprocess import (
    clean_text as clean_oscar_text,
)
from openthaigpt_pretraining_data.core.perplexity import (
    classify_spam,
    sample_text_back,
)
import argparse
import jsonlines
from multiprocessing import Pool
import numpy as np
import scipy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    help='Name of an input file (Default: "scripts/input.jsonl")',
    default="scripts/input.jsonl",
)
parser.add_argument(
    "--output_file",
    help='Name of an output file (Default: "scripts/output.jsonl")',
    default="scripts/output.jsonl",
)
parser.add_argument(
    "--buffer_capacity",
    help="""Size of the buffer.The bigger it is, the more parallelization benefits us
            but be aware of memory consumtion. (Default: 1000)""",
    default=1000,
)
parser.add_argument(
    "--sampled_back_ratio",
    help="""Ratio of data classified as spam to sampled back to the dataset.
    (Default: 0.1)""",
    default=0.1,
)


args = parser.parse_args()


def clean_data(item):
    text = item["text"]

    text = text.strip()
    text = clean_mc4_text(text)

    if text == "":
        item["text"] = text
        return None, None, item

    text = clean_oscar_text(text)

    if text == "":
        item["text"] = text
        return None, None, item

    prediction, log_pp_score = classify_spam(text)

    item["text"] = text
    return prediction, log_pp_score, item


BUFFER_CAP = int(args.buffer_capacity)
idx = 0
if __name__ == "__main__":
    with open(args.input_file, "r", encoding="utf-8") as reader, jsonlines.open(
        args.output_file, "w"
    ) as writer:
        buffer = []
        is_read_all = False
        while True:
            line_content = reader.readline().strip()

            if line_content == "":
                is_read_all = True
            else:
                item = eval(line_content)
                buffer.append(item)

            if len(buffer) != BUFFER_CAP and not is_read_all:
                continue

            with Pool(2) as pool:
                result = pool.map(clean_data, buffer)

            spam_data_points = []
            log_score_list = []
            for prediction, score, data in result:
                # if classifier predict not spam
                if prediction == 0:
                    data["source_id"] = idx
                    writer.write(data)
                    idx += 1
                elif prediction == 1:
                    spam_data_points.append(data)
                    log_score_list.append(score)

            log_scores = np.array(log_score_list)
            mean = np.mean(log_scores)
            std = np.mean(log_scores)

            probs = scipy.stats.norm.pdf(
                log_scores,
                loc=mean,
                scale=std,
            )

            # sampled some text classified as spam back
            sampled_back_texts = sample_text_back(
                spam_data_points,
                probs,
                percentage=float(args.sampled_back_ratio),
            )
            for data in sampled_back_texts:
                data["source_id"] = idx
                writer.write(data)
                idx += 1
            if is_read_all:
                break
            buffer = []
