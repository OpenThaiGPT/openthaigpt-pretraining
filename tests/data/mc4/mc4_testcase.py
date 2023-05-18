# flake8: noqa
from headers.test_texts import (
    none_character_text,
    none_tone_mark_text,
    gamble_text,
    football_text,
    hotel_ad_text,
    sale_text,
    rent_text,
    json_text,
    javascript_text,
    garbage_text,
    ghost_text,
    url_text,
    menu1_text,
    menu2_text,
    menu3_text,
    menu4_text,
    hashtag_text,
    page_text,
    sidebar_text,
    markup_text,
    embedded_server_text,
    u_text,
    iframe_text,
    block_text,
    email_text,
    ip_text,
    tel_text,
    date_text,
    html_text,
    hex_text,
    refine_text,
    dedup_text,
    output,
)
from headers.pattern import (
    toolarge_re,
    nonechar_re,
    none_tone_mark_re,
    gamble_re,
    football_re,
    hotel_ad_re,
    sale_url_re,
    sale_skip_re,
    sale_re,
    rent_skip_re,
    rent_re,
    json_re,
    script_re,
    garbage_re,
    ghost_re,
    url_re,
    menu1_re,
    menu2_re,
    menu3_re,
    menu4_re,
    hashtag_re,
    page_re,
    sidebar_re,
    markup_re,
    embedded_server_re,
    u_re,
    iframe_re,
    block_re,
    email_re,
    ip_re,
    tel_re,
    date1_re,
    date2_re,
    html_re,
    hex_re,
    refine1_re,
    refine2_re,
    refine3_re,
    refine4_re,
    refine5_re,
    refine6_re,
    refine7_re,
    refine8_re,
    refine9_re,
    refine10_re,
    refine11_re,
    refine12_re,
    refine13_re,
    refine14_re,
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
        print(none_character_text)
        matches = nonechar_re.findall(none_character_text)[:25]
        print("\nremove row=", True if len(matches) == 25 else False)
    elif n == 1:
        print(none_tone_mark_text)
        matches = none_tone_mark_re.findall(none_tone_mark_text)[:25]
        print("\nremove row=", True if len(matches) == 25 else False)
    elif n == 2:
        print(gamble_text)
        matches = gamble_re.findall(gamble_text)[:2]
        print("\nremove row=", True if len(matches) == 2 else False)
    elif n == 3:
        print(football_text)
        matches = football_re.findall(football_text)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 4:
        print(hotel_ad_text)
        matches = hotel_ad_re.findall(hotel_ad_text)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 5:
        print(sale_text)
        matches = sale_re.findall(sale_text)[:3]
        print("\nremove row=", True if len(matches) == 3 else False)
    elif n == 6:
        print(rent_text)
        matches = rent_re.findall(rent_text)[:2]
        print("\nremove row=", True if len(matches) == 2 else False)
    elif n == 7:
        print(json_text)
        matches = json_re.findall(json_text)[:20]
        print("\nremove row=", True if len(matches) == 20 else False)
    elif n == 8:
        print(javascript_text)
        matches = script_re.findall(javascript_text)[:10]
        print("\nremove row=", True if len(matches) == 10 else False)
    elif n == 9:
        print(garbage_text)
        matches = garbage_re.findall(garbage_text)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 10:
        print(ghost_text)
        matches = ghost_re.findall(ghost_text)[:4]
        print("\nremove row=", True if len(matches) == 4 else False)
    elif n == 11:
        print(url_text)
        print(output + url_re.sub(" ", url_text))
    elif n == 12:
        print(menu1_text + "\n" + menu2_text + "\n" + menu3_text + "\n" + menu4_text)
        text1 = menu1_re.sub(" ", menu1_text)
        text2 = menu2_re.sub(" ", menu2_text)
        text3 = menu3_re.sub(" ", menu3_text)
        text4 = menu4_re.sub(" ", menu4_text)
        print(output + text1 + "\n" + text2 + "\n" + text3 + "\n" + text4)
    elif n == 13:
        print(hashtag_text)
        print(output + hashtag_re.sub(" ", hashtag_text))
    elif n == 14:
        print(page_text)
        print(output + page_re.sub(" ", page_text))
    elif n == 15:
        print(sidebar_text)
        print(output + sidebar_re.sub(" ", sidebar_text))
    elif n == 16:
        print(markup_text)
        print(output + markup_re.sub(" ", markup_text))
    elif n == 17:
        print(embedded_server_text)
        print(output + embedded_server_re.sub(" ", embedded_server_text))
    elif n == 18:
        print(u_text)
        print(output + u_re.sub(" ", u_text))
    elif n == 19:
        print(iframe_text)
        print(output + iframe_re.sub(" ", iframe_text))
    elif n == 20:
        print(block_text)
        print(output + block_re.sub(" ", block_text))
    elif n == 21:
        print(email_text)
        print(output + email_re.sub(" ", email_text))
    elif n == 22:
        print(ip_text)
        print(output + ip_re.sub(" ", ip_text))
    elif n == 23:
        print(tel_text)
        print(output + tel_re.sub(" ", tel_text))
    elif n == 24:
        print(date_text)
        text = date1_re.sub(" ", date_text)
        text = date2_re.sub(" ", text)
        print(output + text)
    elif n == 25:
        print(html_text)
        print(output + html_re.sub(" ", html_text))
    elif n == 26:
        print(hex_text)
        print(output + hex_re.sub(" ", hex_text))
    elif n == 27:
        print(refine_text)
        text = refine1_re.sub(" ", refine_text)
        text = refine2_re.sub(" ", text)
        text = refine3_re.sub(" ", text)
        text = refine4_re.sub(" ", text)
        text = refine5_re.sub(" ", text)
        text = refine6_re.sub(" ", text)
        text = refine7_re.sub(" ", text)
        text = refine8_re.sub(" ", text)
        text = refine9_re.sub(" ", text)
        text = refine10_re.sub(" ", text)
        text = refine11_re.sub(" ", text)
        text = refine12_re.sub(" ", text)
        text = refine13_re.sub(" ", text)
        text = refine14_re.sub(" ", text)
        text = "\n".join(line for line in text.split("\n") if len(line) > 30)
        print(output + text)
    elif n == 28:
        print(dedup_text)
        # Split the text into lines and remove any empty lines
        lines = [line for line in dedup_text.split("\n") if line]
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
        print(output + text)

    input("Press any key --->")
