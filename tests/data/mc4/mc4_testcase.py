# flake8: noqa
from headers.test_texts import (
    NONE_CHARACTER_TEXT,
    NONE_TONE_MARK_TEXT,
    GAMBLE_TEXT,
    FOOTBALL_TEXT,
    HOTEL_AD_TEXT,
    SALE_TEXT,
    RENT_TEXT,
    JSON_TEXT,
    JAVASCRIPT_TEXT,
    GARBAGE_TEXT,
    GHOST_TEXT,
    URL_TEXT,
    MENU1_TEXT,
    MENU2_TEXT,
    MENU3_TEXT,
    MENU4_TEXT,
    HASHTAG_TEXT,
    PAGE_TEXT,
    SIDEBAR_TEXT,
    MARKUP_TEXT,
    EMBEDDED_SERVER_TEXT,
    U_TEXT,
    IFRAME_TEXT,
    BLOCK_TEXT,
    EMAIL_TEXT,
    IP_TEXT,
    TEL_TEXT,
    DATE_TEXT,
    HTML_TEXT,
    HEX_TEXT,
    REFINE_TEXT,
    DEDUP_TEXT,
    OUTPUT,
)
from openthaigpt_pretraining_data.mc4.pattern import (
    NONECHAR_RE,
    NONE_TONE_MARK_RE,
    GAMBLE_RE,
    FOOTBALL_RE,
    HOTEL_AD_RE,
    SALE_RE,
    RENT_RE,
    JSON_RE,
    SCRIPT_RE,
    GARBAGE_RE,
    GHOST_RE,
    URL_RE,
    MENU1_RE,
    MENU2_RE,
    MENU3_RE,
    MENU4_RE,
    HASHTAG_RE,
    PAGE_RE,
    SIDEBAR_RE,
    MARKUP_RE,
    EMBEDDED_SERVER_RE,
    U_RE,
    IFRAME_RE,
    BLOCK_RE,
    EMAIL_RE,
    IP_RE,
    TEL_RE,
    DATE1_RE,
    DATE2_RE,
    HTML_RE,
    HEX_RE,
    REFINE1_RE,
    REFINE2_RE,
    REFINE3_RE,
    REFINE4_RE,
    REFINE5_RE,
    REFINE6_RE,
    REFINE7_RE,
    REFINE8_RE,
    REFINE9_RE,
    REFINE10_RE,
    REFINE11_RE,
    REFINE12_RE,
    REFINE13_RE,
    REFINE14_RE,
)
import re

while True:
    print("\nTest mC4 clening cases")
    print("----------------------")
    print("Select a test below:")
    print(" 0. none_char")
    print(" 1. none_tone_mask")
    print(" 2. gamble")
    print(" 3. football")
    print(" 4. hotel_ad")
    print(" 5. sale")
    print(" 6. rent")
    print(" 7. json patterns")
    print(" 8. script patterns")
    print(" 9. garbage patterns")
    print("10. ghost patterns")
    print("11. url")
    print("12. menu")
    print("13. hashtag")
    print("14. pagination")
    print("15. sidebar")
    print("16. markup language")
    print("17. embeded server-side code")
    print("18. U+XXXX")
    print("19. iframe")
    print("20. block patterns")
    print("21. email")
    print("22. ip address")
    print("23. telephone")
    print("24. date patterns")
    print("25. html")
    print("26. hex")
    print("27. refinements")
    print("28. deduplicate common-prefix")
    print()

    quitFlag = False
    items = [i for i in range(0, 29)]
    while True:
        try:
            n = int(input("Select (99=Quit):"))
        except ValueError:
            print("That's not a valid number. Try again.")
        if n == 99:
            quitFlag = True
            break
        if n in items:
            break
    if quitFlag:
        break

    if n == 0:
        print(NONE_CHARACTER_TEXT)
        matches = NONECHAR_RE.findall(NONE_CHARACTER_TEXT)[:25]
        print("\nremove row=", True if len(matches) == 25 else False)
    elif n == 1:
        print(NONE_TONE_MARK_TEXT)
        matches = NONE_TONE_MARK_RE.findall(NONE_TONE_MARK_TEXT)[:25]
        print("\nremove row=", True if len(matches) == 25 else False)
    elif n == 2:
        print(GAMBLE_TEXT)
        matches = GAMBLE_RE.findall(GAMBLE_TEXT)[:2]
        print("\nremove row=", True if len(matches) == 2 else False)
    elif n == 3:
        print(FOOTBALL_TEXT)
        matches = FOOTBALL_RE.findall(FOOTBALL_TEXT)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 4:
        print(HOTEL_AD_TEXT)
        matches = HOTEL_AD_RE.findall(HOTEL_AD_TEXT)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 5:
        print(SALE_TEXT)
        matches = SALE_RE.findall(SALE_TEXT)[:3]
        print("\nremove row=", True if len(matches) == 3 else False)
    elif n == 6:
        print(RENT_TEXT)
        matches = RENT_RE.findall(RENT_TEXT)[:2]
        print("\nremove row=", True if len(matches) == 2 else False)
    elif n == 7:
        print(JSON_TEXT)
        matches = JSON_RE.findall(JSON_TEXT)[:20]
        print("\nremove row=", True if len(matches) == 20 else False)
    elif n == 8:
        print(JAVASCRIPT_TEXT)
        matches = SCRIPT_RE.findall(JAVASCRIPT_TEXT)[:10]
        print("\nremove row=", True if len(matches) == 10 else False)
    elif n == 9:
        print(GARBAGE_TEXT)
        matches = GARBAGE_RE.findall(GARBAGE_TEXT)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 10:
        print(GHOST_TEXT)
        matches = GHOST_RE.findall(GHOST_TEXT)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 11:
        print(URL_TEXT)
        print(OUTPUT + URL_RE.sub(" ", URL_TEXT))
    elif n == 12:
        print(MENU1_TEXT + "\n" + MENU2_TEXT + "\n" + MENU3_TEXT + "\n" + MENU4_TEXT)
        text1 = MENU1_RE.sub(" ", MENU1_TEXT)
        text2 = MENU2_RE.sub(" ", MENU2_TEXT)
        text3 = MENU3_RE.sub(" ", MENU3_TEXT)
        text4 = MENU4_RE.sub(" ", MENU4_TEXT)
        print(OUTPUT + text1 + "\n" + text2 + "\n" + text3 + "\n" + text4)
    elif n == 13:
        print(HASHTAG_TEXT)
        print(OUTPUT + HASHTAG_RE.sub(" ", HASHTAG_TEXT))
    elif n == 14:
        print(PAGE_TEXT)
        print(OUTPUT + PAGE_RE.sub(" ", PAGE_TEXT))
    elif n == 15:
        print(SIDEBAR_TEXT)
        print(OUTPUT + SIDEBAR_RE.sub(" ", SIDEBAR_TEXT))
    elif n == 16:
        print(MARKUP_TEXT)
        print(OUTPUT + MARKUP_RE.sub(" ", MARKUP_TEXT))
    elif n == 17:
        print(EMBEDDED_SERVER_TEXT)
        print(OUTPUT + EMBEDDED_SERVER_RE.sub(" ", EMBEDDED_SERVER_TEXT))
    elif n == 18:
        print(U_TEXT)
        print(OUTPUT + U_RE.sub(" ", U_TEXT))
    elif n == 19:
        print(IFRAME_TEXT)
        print(OUTPUT + IFRAME_RE.sub(" ", IFRAME_TEXT))
    elif n == 20:
        print(BLOCK_TEXT)
        print(OUTPUT + BLOCK_RE.sub(" ", BLOCK_TEXT))
    elif n == 21:
        print(EMAIL_TEXT)
        print(OUTPUT + EMAIL_RE.sub(" ", EMAIL_TEXT))
    elif n == 22:
        print(IP_TEXT)
        print(OUTPUT + IP_RE.sub(" ", IP_TEXT))
    elif n == 23:
        print(TEL_TEXT)
        print(OUTPUT + TEL_RE.sub(" ", TEL_TEXT))
    elif n == 24:
        print(DATE_TEXT)
        text = DATE1_RE.sub(" ", DATE_TEXT)
        text = DATE2_RE.sub(" ", text)
        print(OUTPUT + text)
    elif n == 25:
        print(HTML_TEXT)
        print(OUTPUT + HTML_RE.sub(" ", HTML_TEXT))
    elif n == 26:
        print(HEX_TEXT)
        print(OUTPUT + HEX_RE.sub(" ", HEX_TEXT))
    elif n == 27:
        print(REFINE_TEXT)
        text = REFINE1_RE.sub(" ", REFINE_TEXT)
        text = REFINE2_RE.sub(" ", text)
        text = REFINE3_RE.sub(" ", text)
        text = REFINE4_RE.sub(" ", text)
        text = REFINE5_RE.sub(" ", text)
        text = REFINE6_RE.sub(" ", text)
        text = REFINE7_RE.sub(" ", text)
        text = REFINE8_RE.sub(" ", text)
        text = REFINE9_RE.sub(" ", text)
        text = REFINE10_RE.sub(" ", text)
        text = REFINE11_RE.sub(" ", text)
        text = REFINE12_RE.sub(" ", text)
        text = REFINE13_RE.sub(" ", text)
        text = REFINE14_RE.sub(" ", text)
        text = "\n".join(line for line in text.split("\n") if len(line) > 30)
        print(OUTPUT + text)
    elif n == 28:
        print(DEDUP_TEXT)
        # Split the text into lines and remove any empty lines
        lines = [line for line in DEDUP_TEXT.split("\n") if line]
        # Initialize the list with the first line
        deduplicated_list = [lines[0]]
        # Iterate over the rest of the lines
        for i in range(1, len(lines)):
            # Find the common prefix between this line and the previous line
            common_prefix = ""
            for char1, char2 in zip(lines[i], lines[i - 1]):
                if char1 == char2:
                    common_prefix += char1
                else:
                    break
            # Remove the common prefix from this line and add it to the list
            deduplicated_list.append(lines[i][len(common_prefix) :])
        text = "\n".join(deduplicated_list)
        print(OUTPUT + text)

    input("Press any key --->")
