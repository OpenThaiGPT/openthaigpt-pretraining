from openthaigpt_pretraining_data.merge_pdf import (
    read_pdf_all_pages,
    clean_dup_esc_char,
    clean_by_rules,
    clean_by_char_structure,
)
import jsonlines
import glob  # type: ignore
import tqdm
import hydra

NULL = "null"
TEXT_KEY = "text"
META_KEY = "meta"
SOURCE_KEY = "source"
SOURCE_ID_KEY = "source_id"
CREATED_DATE_KEY = "create_date"
UPDATED_DATE_KEY = "update_date"


@hydra.main(version_base=None, config_path="./config", config_name="crawl_set_annual")
def main(cfg):
    config = cfg.convert_to_jsonl

    pdf_path = config.pdf_path
    text_rule_file = config.text_rule_file
    output_path = config.output_path
    paths = glob.glob(f"{pdf_path}/***/**/*.PDF")

    with jsonlines.open(output_path, "w") as writer:
        for file_path in tqdm.tqdm(paths):
            split_file_name = file_path.split("\\")

            raw_text = read_pdf_all_pages(file_path)
            clean_text = clean_dup_esc_char(raw_text)
            clean_text = clean_by_rules(clean_text, text_rule_file)
            clean_text = clean_by_char_structure(clean_text)

            data = {
                TEXT_KEY: clean_text,
                SOURCE_KEY: split_file_name[-3],
                SOURCE_ID_KEY: NULL,
                CREATED_DATE_KEY: split_file_name[-2],
                UPDATED_DATE_KEY: split_file_name[-2],
                META_KEY: NULL,
            }

            writer.write(data)


if __name__ == "__main__":
    main()  # type: ignore
