from openthaigpt_pretraining_data.merge_pdf import pdf_2_text_markup
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

SOURCE = "admincourt"


@hydra.main(version_base=None, config_path="./config", config_name="crawl_admincourt")
def main(cfg):
    config = cfg.convert_to_jsonl

    pdf_path = config.pdf_path
    text_rule_file = config.text_rule_file
    output_path = config.output_path
    paths = glob.glob(f"{pdf_path}/*.PDF")

    with jsonlines.open(output_path, "w") as writer:
        for file_path in tqdm.tqdm(paths):
            file_name = file_path.replace("\\", "/").split("/")[-1]
            split_file_name = file_name.split("+")

            clean_text = pdf_2_text_markup(file_path, text_rule_file)

            data = {
                TEXT_KEY: clean_text,
                SOURCE_KEY: SOURCE,
                SOURCE_ID_KEY: split_file_name[1],
                CREATED_DATE_KEY: split_file_name[0],
                UPDATED_DATE_KEY: split_file_name[0],
                META_KEY: NULL,
            }

            writer.write(data)


if __name__ == "__main__":
    main()  # type: ignore
