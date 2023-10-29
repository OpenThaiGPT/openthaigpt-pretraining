# ThaiGOV Dataset Crawler

This script access https://www.thaigov.go.th website and crawl all news from it.

## Instruction
1. Run `pip install ./src/data` and `pip install ./src/core`
2. dvc pull `Thaigov_crawl_lists.csv`
3. edit `Thaigov_crawl_lists.csv` to point to the latest news index
4. Run `python src/data/scripts/crawl_thaigov/nuch_crawl_thaigov_all.py`
5. The result will be in `test_crawl_lists.csv`

## Characteristic of the data
Thai Government news website will be reset when the new government is formed, so please make sure to run the pipeline before the new government is form.