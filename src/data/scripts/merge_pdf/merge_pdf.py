from openthaigpt_pretraining_data.merge_pdf import (
    pdf_2_text_markup,
)

import hydra


@hydra.main(version_base=None, config_path="./config", config_name="merge_pdf")
def main(cfg):
    pdf_2_text_markup(cfg.pdf_file, cfg.text_rule_file)


if __name__ == "__main__":
    main()  # type: ignore
