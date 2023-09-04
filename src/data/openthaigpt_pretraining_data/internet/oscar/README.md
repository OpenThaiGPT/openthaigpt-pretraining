# OSCAR preprocessing code

This folder contains codes and regex patterns which are the result from the study of OSCAR dataset (2019, 2021, 2022, 2023).

The regex patterns are compiled in `keywords.py` and import to use in `preprocess.py`

The `notebook` folder contains the experiment, observation and EDA notebooks for OSCAR.

## How does the code work ?

1. Check the garbage keywords in each text. If the text contains any of keywords, remove the text.

| Keywords                  |
|--------------------------|
| Pornography              |
| Gamble                   |
| Spam movie website       |
| Spam like ads            |
| Some programming code    |
| Some webboard keywords   |

2. If the text is not removed by step 1, Check the less garbage keywords in each text. If the ratio of the keywords length and the text length are above the thresholds, remove the text

| Keywords                               | Thresholds |
|----------------------------------------|------------|
| Thai month names                        | 0.015      |
| Some programming code related symbol   | 0.075      |
| Space                                  | 0.13       |
| Comma                                  | 0.05       |
| Thai character                          | 0.5        |

3. If the text is not removed by step 1 and 2, less severe patterns will be check and remove partially.

| Patterns                             |
|-------------------------------------|
| Webboard special characters («»)     |
| Webboard specific keywords           |
| Browser related keywords ("cookie setting") |
| Other spam ("นโยบายความเป็นส่วนตัว", "หัวข้อ:") |

## Running

The code is imported in `src/data/scripts/internet` and you can use it together with mc4 code there. 

This code is not meant to be run directly. If you want to run with your custom logic please create folder in `src/data/scripts` and import the function you want.

## Note

- The keywords and patterns are from an observation on the sample of each OSCAR dataset.
