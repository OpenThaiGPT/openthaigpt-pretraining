# encoding: utf-8
# flake8: noqa
import re

#### Gamble Clean Words
gamble_words = [
    "พนัน",
    "แทงบอล",
    "แทง",
    "บาคารา",
    "บา คา รา",
    "เกมพนัน",
    "คาสิโน",
    "คา สิ โน",
    "หวย",
    "สล็อต",
    "กาสิโน",
    "casino",
    "slot",
    "เลขเด็ด",
    "สูตรหวย",
    "a s i n o",
    "sbobet",
    "fun88",
    "ufabet",
    "บาคาร่า",
    "บา คา ร่า",
    "รูเล็ต",
    "ทำนายฝัน",
    "เลขเด่น",
    "สรุปผลบอล",
    "ไฮไลท์ฟุตบอล",
    "วิเคราะห์บอล",
    "ดูบอลสด",
    "พรีเมียร์ลีก",
    "บอลประจำวัน",
    "บอลเต็ง",
    "บอลเด็ด",
    "องค์ลงรวย",
    "สูตรปลาตะเพียน",
    "สามตัวตรง",
    "วิเคราะห์ข้อมูลล่าง",
    "ต่อ ครึ่งลูก",
    "ครึ่งลูกลบ",
    "เสมอควบครึ่ง",
    "ครึ่งควบลูก",
]

#### Sale Clean Words
sale_skip_words = [
    "สอบราคา",
    "จัดซื้อจัดจ้าง",
    "ชมรม",
    "สมาคม",
    "นักลงทุน",
    "นักการตลาด",
    "ของกลาง",
    "การลงทุน",
    "นักวิเคราะห์",
    "ขายให้แก่ประชาชน",
    "การลดต้นทุน",
    "การเสนอราคา",
    "กระทรวง",
    "ตลาดหลักทรัพย์",
    "ยอดขายไม่ดี",
    "ยอดขายไม่ค่อยดี",
    "ผู้ประกอบการธุรกิจ",
    "ออกใบอนุญาต",
    "ผู้ประกอบกิจการ",
]
sale_url_words = [
    "alibaba.com",
    "shopee.co.th",
    "lazada.com",
    "DocPlayer.net",
    "Alibaba",
    "AliExpress",
    "Aliexpress",
    "TripAdvisor",
    "jobbkk.com",
]
sale_words = [
    "ขาย",
    "ซ่อม",
    "ราคา",
    "มือสอง",
    "เช่า",
    "ครีม",
    "ฝ้ากระ",
    "จุดด่างดำ",
    "รับส่วนลด",
    "โปรโมชั่น",
    "กวดวิชา",
    "ติวเตอร์",
    "SEO",
    "คอร์สเรียน SEO",
    "จำหน่าย",
    "ทัวร์",
    "สินค้ามาใหม่",
    "สินค้าทั้งหมด",
    "รีวิวสินค้า",
    "เคสกันกระแทก",
    "ประกาศ",
    "ลงขายของ",
    "เลือกขนาด",
    "บริการจัดส่ง",
    "จัดอันดับ",
    "คาราโอเกะ",
    "จำหน่าย",
    "หาเงินออนไลน์",
    "สั่งซื้อ",
    "ลดกระหนำ่",
    "รหัส",
    "ลงประกาศฟรี",
    "หยิบใส่ตะกร้า",
    "สนใจ",
    "ซื้อ",
    "สินค้า",
    "ผลิตภัณฑ์",
]

#### Rent Clean Words
rent_skip_words = [
    "สอบราคา",
    "จัดซื้อจัดจ้าง",
    "ชมรม",
    "สมาคม",
    "นักลงทุน",
    "นักการตลาด",
    "ของกลาง",
    "การลงทุน",
    "นักวิเคราะห์",
    "ขายให้แก่ประชาชน",
    "การลดต้นทุน",
    "การเสนอราคา",
    "กระทรวง",
    "ตลาดหลักทรัพย์",
]
rent_words = [
    "บ้านมือสอง",
    "ให้เช่า",
    "เช่า",
    "บ้านเดี่ยว",
    "อพาร์ทเม้นท์",
    "อสังหาริมทรัพย์",
    "เพนท์เฮ้าส์",
    "ทาวน์เฮ้าส์",
]

#### Script Clean Words
script_words = [
    "function",
    "var",
    "click",
    "margin",
    "width",
    "height",
    "return",
    "else",
    "alert",
    "<br>",
    "href",
]

#### Garbage Clean Words
garbage_words = [
    "โหงวเฮ้ง",
    "ครีมฟอกสี",
    "ครีมผิวขาว",
    "ฟอกสี",
    "ไวท์เทนนิ่งครีม",
    "ครีมไวท์เทนนิ่ง",
    "ครีมลบฝ้ากระ",
    "รับสร้างบ้าน",
    "ครีมโรคสะเก็ดเงิน",
    "บริการจองตั๋ว",
    "บริการรีดผ้า",
    "อาหารเสริมลดน้ำหนัก",
    "ยาลดน้ำหนัก",
    "ลดไขมัน",
    "ผิงโซดา",
    "สร้างบ้าน",
    "ช่างกุญแจ",
    "ช่างโลหะ",
    "ช่างโยธา",
    "ช่างเครื่องยนต์",
    "ช่างไม้",
    "ช่างกลโรงงาน",
    "ช่างไฟฟ้า",
    "ปรสิต",
    "หนอน",
    "เวิร์ม",
]

#### Football teams
football_teams = [
    "ยูเวนตุส",
    "อินเตอร์ มิลาน",
    "นาโปลี",
    "เอซี มิลาน",
    "ลาซิโอ",
    "โรม่า",
    "กัลโซ่",
    "เซเรีย",
    "ปาร์ม่า",
    "เอฟเวอร์ตัน",
    "ซันเดอร์แลนด์",
    "ลิเวอร์พูล",
    "แมนเชสเตอร์",
    "นิวคาสเซิล",
    "เชลซี",
    "อาร์เซนอล",
    "คลิสตัลพาเลช",
    "เซาแทมป์ตัน",
    "เซาแธมป์ตัน",
    "เชฟฟิลด์",
    "ฟอเรสต์",
    "เบอร์ตัน",
    "เบรนท์ฟอร์ด",
    "ฟูแล่ม",
    "ไฮไลท์ฟุตบอล",
    "เลบันเต้",
    "บาร์เซโลน่า",
    "เรอัล มาดริด",
    "เอสปันญ่อล",
]

#### Hotels Advertising
hotel_ad = [
    "โรงแรมอันดับ",
    "ที่พักแบบพิเศษอันดับ",
    "สถานที่พักอันดับ",
    "สถานที่พักคุ้มค่าอันดับ",
    "โรงแรมใกล้กับ",
    "โรงแรมที่ใกล้",
    "โรงแรม 4 ดาว",
    "โรงแรม 3 ดาว",
    "ที่พักพร้อมอาหารเช้า",
    "โรงแรมราคาถูก",
    "โรงแรมหรู",
]

#########
# PRE-COMPILE REGEX to object for speed up processing.
#########
# -----------------------------------------------------
# Remove useless row that make overhead in regex processing

# Unusual row - line size too large
# if there are 3 large lines ( 500 characters each)
toolarge_line_pattern = ".{1500}"
toolarge_re = re.compile(toolarge_line_pattern, re.MULTILINE)

nonechar_pattern = "๮|๞|๨|๡|๷|๻|๫|͹"
nonechar_re = re.compile(nonechar_pattern, re.MULTILINE)

none_tone_mark_pattern = "ก าหนด|เป าหมาย|พ ฒนา|ค ณภาพ|ว จ ย|ค ณล กษณะ|ต างๆ|เป น |ให |บร หาร|ปร บปร ง|ใหม|อย าง|เง น"
none_tone_mark_re = re.compile(none_tone_mark_pattern, re.MULTILINE)

# -----------------------------------------------------

gamble_pattern = "|".join(gamble_words)
gamble_re = re.compile(gamble_pattern, re.MULTILINE)

football_pattern = "|".join(football_teams)
football_re = re.compile(football_pattern, re.MULTILINE)

hotel_ad_pattern = "|".join(hotel_ad)
hotel_ad_re = re.compile(hotel_ad_pattern, re.MULTILINE)

sale_url_pattern = "|".join(sale_url_words)
sale_url_re = re.compile(sale_url_pattern, re.MULTILINE)
sale_skip_pattern = "|".join(sale_skip_words)
sale_skip_re = re.compile(sale_skip_pattern, re.MULTILINE)
sale_pattern = "|".join(sale_words)
sale_re = re.compile(sale_pattern, re.MULTILINE)

rent_skip_pattern = "|".join(rent_skip_words)
rent_skip_re = re.compile(rent_skip_pattern, re.MULTILINE)
rent_pattern = "|".join(rent_words)
rent_re = re.compile(rent_pattern, re.MULTILINE)

json_pattern = r"\s*\"(?:\w)*\"\s*:"
json_re = re.compile(json_pattern, re.MULTILINE)

script_pattern = r"\b" + "|".join(script_words) + r"\b"
script_re = re.compile(script_pattern, re.MULTILINE)

garbage_pattern = "|".join(garbage_words)
garbage_re = re.compile(garbage_pattern, re.MULTILINE)

ghost_pattern = "เธฃเน|เธเธญ|เธเน|เธฐเธ|เธฅเธฐ|เธซเธฒ|เธญเธฒ|เธดเธ|เธตเธข|เธญเน|เธญเธ|เธดเน|เธฑเธ|เธกเน|เธฒเธ|เธชเน|เน€เธ"
ghost_re = re.compile(ghost_pattern, re.MULTILINE)

url_pattern = r"\b(?:(?:https?|ftp)://[^\s/$\.\?#].[^\s]*)\b|\b(?:www\.?)?(?:(?:[\w-]*)\.)*(?:com|net|org|info|biz|me|io|co|asia|xyz|th|cn|in|uk|jp|ru)\b"
# url_pattern = r'\\b(?:(?:https?|ftp)://[^\s/$\.\?#].[^\s]*)\\b|\\b(?:www\.?)?(?:(?:[\w-]*)\.)*(?:com|net|org|info|biz|me|io|co|asia|xyz|th|cn|in|uk|jp|ru)\\b'
url_re = re.compile(url_pattern, re.MULTILINE)

menu1_pattern = "\|(?:[^\|\n]*\|)+.*"
menu1_re = re.compile(menu1_pattern, re.MULTILINE)

menu2_pattern = "\|(?:[^\|\n]*\|)+"
menu2_re = re.compile(menu2_pattern, re.MULTILINE)

menu3_pattern = "(?:(?:[^/\n]*/){4,}.*)"
menu3_re = re.compile(menu3_pattern, re.MULTILINE)

menu4_pattern = "[^\n]{0,20}[ ]{0,2}[>»\\\\].*"
menu4_re = re.compile(menu4_pattern, re.MULTILINE)

hashtag_pattern = "#\d*[ ].{0,300}|#(?:(?:[^ \n]*)[ ]?)+|Tag Archives[ ]{0,2}:.{0,300}|Posts Tagged[ ]{0,2}:.{0,300}|HASTAG[ ]{0,2}:.{0,300}|Tag[s]?[ ]{0,2}:.{0,300}|Tagged[ ].{0,300}"
hashtag_re = re.compile(hashtag_pattern, re.MULTILINE)

page_pattern = "(?:<<[ ])?(?:ก่อนหน้า|ย้อนกลับ)[ ]{0,2}(?:\[[ ]?\d{0,6}[ ]?\]|[ ]?\d{0,6}[ ]?)*(?:ต่อไป|หน้าถัดไป|ถัดไป)?(?:[ ]?>>)?|<<(?:[ ]\d{0,6}[ ]\-[ ]\d{0,6})+[ ].{0,100}"
page_re = re.compile(page_pattern, re.MULTILINE)

sidebar_pattern = ".{0,40}(?:(?:\[|\()\d{0,9}(?:\]|\))(?:[ ]{0,2})?,?)"
sidebar_re = re.compile(sidebar_pattern, re.MULTILINE)

markup_pattern = "\{\{[^\}]*\}\}|\{\{.*"
markup_re = re.compile(markup_pattern, re.MULTILINE)

embedded_server_pattern = "<%[ ]*[^%]*%>|<%.*"
embedded_server_re = re.compile(embedded_server_pattern, re.MULTILINE)

u_pattern = "\uFEFF|\u00AD|[\u200A-\u200F]|\uFFFD|[\uE000-\uF8FF]|[\u202A-\u202C]|\u0092|[\u0091-\u0096]|\u2028|\u2066|\u2069|\u008d|\u0081|\u008E|<U\+[0-9A-Fa-f]{4}>"
u_re = re.compile(u_pattern, re.MULTILINE)

iframe_pattern = "<iframe.*?<\/iframe>\s*|<iframe.*"
iframe_re = re.compile(iframe_pattern, re.MULTILINE)

block_pattern = "(?:\[[^\]]*\])|(?:«[^»]*»)|(?:<<([^>]*)>>)"
block_re = re.compile(block_pattern, re.MULTILINE)

email_pattern = "(?:(?:([Ee]?mail|อีเมล์)[ ]{0,2}:?[ ]{0,5})?)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
email_re = re.compile(email_pattern, re.MULTILINE)

ip_pattern = "\((?:(?:X{1,3}|\d{1,3})\.){3}(?:X{1,3}|\d{1,3})\)|\(?IP:?[ ]?(?:(?:X{1,3}|\d{1,3})\.){3}(?:X{1,3}|\d{1,3})\)?"
ip_re = re.compile(ip_pattern, re.MULTILINE)

tel_pattern = "(?:(?:[Pp]hone|[Mm]obile|มือถือ|Tel|TEL|Fax|FAX|เบอร์โทรศัพท์|เลขโทรศัพท์|เบอร์ติดต่อ|โทรศัพท์|โทรสาร[ ]{0,2}:|เบอร์โทร|โทร[ ]{0,2}:|โทร\.|โทร[ ]|ติดต่อที่[ ]{0,2}:?|ติดต่อ[ ]{0,2}:?)[ ]{0,2}):?(?:(?:[ ]{0,2})?(?:(?:\d{3}-\d{7})|(?:\d{4}-\d{6})|(?:\d{3}-\d{3}-\d{4}|(?:\d{3}-\d{3}-\d{3})|(?:\d{1}-\d{4}-\d{4})|(?:\d{2}-\d{3}-\d{4})|(?:\d{2}\s\d{3}\s\d{4})|(?:\d{2}-\d{7})|(?:\d{3}\s\d{3}\s\d{4})|(?:\d{3}\s\d{3}\s\d{3})|(?:\d{10})))[ ]{0,2},?)+|02\d{7}|0[3-7][2-9]\d{6}|0[6-9][0-9]\d{7}"
tel_re = re.compile(tel_pattern, re.MULTILINE)

date1_pattern = "(?:(?:การปรับปรุงปัจจุบัน|ตั้งแต่|ลงประกาศเมื่อ|อัพเดทล่าสุด|แก้ไขครั้งสุดท้าย|แก้ไขครั้งล่าสุด|เผยแพร่เมื่อ|เผยแพร่|เขียนเมื่อ|ตอบเมื่อ|เมื่อ|เขียนวันที่|วันที่|วัน)?(?:[ ]{0,2}:[ ]{0,2})?(?:จันทร์|อังคาร|พุธ|พฤหัสบดี|พฤหัสฯ?\.?|ศุกร์|เสาร์|อาทิตย์|จ\.|อ\.|พ\.|พฤ\.|ศ\.|ส\.|อา\.?)?(?:[ ]{0,2}ที่)?(?:[ ]{0,2}[\w\u0E01-\u0E5B]*[ ]{0,2}(?:,|(?:-|\u2013)))?(?:\d{1,4}[ ]{0,2}-)?[ ]{0,2}\d{0,4}(?:-|[ ]{0,2})(?:เดือน[ ]{0,2})?(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.?ค\.?|ก\.?พ\.?|มี\.?ค\.?|เม\.?ย\.?|พ\.?ค\.?|มิ\.?ย\.?|ก\.?ค\.?|ส\.?ค\.?|ก\.?ย\.?|ต\.?ค\.?|พ\.?ย\.?|ธ\.?ค\.?|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|/)(?:-|[ ]|,]){0,2}(?:[ ]{0,2})?(?:\d{4}|\d{2})?,?(?:(?:[ ]{0,2}\d{4}|\d{2}:\d{2}:\d{2})|(?:[ ]{0,2}\d{1,2}:\d{2}(?::\d{2})?[ ]*(?:(?:(?:p|P|a|A)(?:m|M)))?)|[ ]{0,2}เวลา[ ]{0,2}\d{1,2}:\d{2}:\d{2})?(?:[ ]{0,2}น\.)?(?:[ ]{0,2}:[ ]{0,2}\d{1,2}(?:\.|:)\d{2}[ ]{0,2}น\.)?(?:[ ]{0,2}เวลา[ ]{0,2}\d{1,2}:\d{2}[ ]{0,2}[pPaA][mM][ ]{0,2}(?:PDT)?)?(?:[ ]{0,2}เวลา[ ]{0,2}\d{1,2}(?:\.|:)\d{2}[ ]{0,2}(?:น\.)?)?(?:[ ]*\d*[ ]{0,9}ผู้ชม[ ]{0,2}\d)?(?:[ ]{0,2}เวลา[ ]{0,2}:[ ]{0,2}\d{1,2}:\d{2}:\d{2}[ ](?:น\.)?)?(?:[ ]{0,2}เข้าชม[ ]{0,2}\d*[ ]{0,8}ครั้ง)?(?:[ ]{0,2}ที่[ ]{0,2}\d{1,2}:\d{2}[ ]{0,2}(?:[pPaA][mM])?)?(?:[ ]{0,2}Views:[ ]{0,2}(?:\d{0,3},?){0,3})?(?:[ ]{0,2}\(\d{1,2}:\d{2}[ ](?:น\.)?[ ]{0,2}\)(?:[ ]{0,2}ความคิดเห็น[ ]{0,2}\d)?)?(?:[ ]{0,2}-[ ]{0,2}\d{1,2}:\d{2}[ ]{0,2}(?:น\.)?)?(?:[ ]{0,2}จำนวนเข้าชม:(?:[ ]{0,2}\d{0,9},?)*)?(?:[ ]*พ\.ศ\.[ ]{0,2}\d{4}[ ]{0,2}(?:\d{1,2}:\d{2}[ ]{0,2}(?:น\.)?)?)?(?:(?:\d{0,3},?){0,3}[ ]{0,2}ครั้ง)?(?:[ ]{0,2}-\d{1,2}(?:\.|:)?\d{2}[ ]{0,2}(?:น\.)?)?(?:เวลา[ ]{0,2}\d{1,2}:\d{2}:\d{2})?(?:[ ]{0,2}(?:ถึง|จนถึง))?)(?:,[ ]{0,2})?(?:[ ]{0,2}(?:[pPaA][mM])?)?|(?:(?:[ ]{0,2}(?:\d{4}|\d{2})-\d{1,2}-(?:\d{4}|\d{1,2}))(?:[ ]{0,2},[ ]{0,2})?(?:\d{1,2}(?:\.|:)\d{2}[ ]{0,2}(?:#\d*)?(?:น\.)?)?(?:[ ]{0,2}\d{1,2}:\d{2}:\d{2})?)|(?:(?:เปิดบริการ[ ]{0,2})?(?:เวลา[ ]{0,2}\d{1,2}:\d{2}-\d{1,2}:\d{2}[ ]{0,2}(?:น\.)?))|(?:(?:Time[ ]Online[ ]:[ ]{0,2})?(?:\d{1,2}:\d{2}:\d{2})(?:[ ]{0,2}น\.)?(?:[ ]{0,2}[pPaA][mM])?)|(?:\(\d{1,2}:\d{2}[ ]{0,2}-[ ]{0,2}\d{1,2}:\d{2}(?:[ ]*น\.)?(?:[ ]{0,2}[pPaA][mM])?\))|(?:นี้[ ]{0,2}เวลา[ ]{0,2}\d{1,2}(?:\.|:)\d{2}[ ]{0,2}(?:น\.)?)|(?:\d{1,2}[ ]{0,2}(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม))"
date1_re = re.compile(date1_pattern, re.MULTILINE)
date2_pattern = "[พค]\.?ศ\.?[ ]{0,2}\d{4}|\d{4}[ ]{0,2}เวลา[ ]{0,2}\d{2}:?\.?\d{2}(?:[ ][Pp][Mm])|[พค]\.?ศ\."
date2_re = re.compile(date2_pattern, re.MULTILINE)

html_pattern = (
    "<br>?|&nbsp|{\s*document\..*|SELECT.*FROM.*WHERE.*|<a\s*href=.*|<img\s*src=.*"
)
html_re = re.compile(html_pattern, re.MULTILINE)

hex_pattern = "(?<![A-Za-z0-9\-ก-๙])(?:[0-9A-Fa-f]{2,})(?![0-9A-Za-z\-ก-๙])"
hex_re = re.compile(hex_pattern, re.MULTILINE)

refine1_pattern = "^[ ]?ตอนที่[ ]*\d{0,3}(?:[-–]?\d{0,3})?[ ]{0,2}.{0,100}|^สั่ง.{0,50}บาท|^[ ]?เลขจดแจ้ง[ ]{0,2}.{0,13}|^.{0,100}\.jpg[ ]{0,2}\(.{0,50}|^.{0,20}รายการ|^[ ]?สนใจ[ ]{0,2}.{0,15}โทร[ ]{0,2}.{0,12}|^[ ]?ผู้แสดงความคิดเห็น.{0,60}|^\(.{0,40}[ ]{0,2}\d{0,5}[ ]{0,2}.{0,10}\).{0,200}|^[ ]?ผู้เข้าชมทั้งหมด.{0,30}|^[ ]?ฉบับที่[ ]{0,2}\d{0,7}[^-–]{0,30}-?–?[ ]|^[ ]?โพสต์ที่แชร์โดย.{0,200}|^[ ]?Copyright.{0,200}|กำลังแสดงหน้าที.{0,200}|[ ]{0,2}รีวิว.{0,100}|^[ ]?ข้อที่ \d{0,4}|^เข้าชม/ผู้ติดตาม.{0,13}"
refine1_re = re.compile(refine1_pattern, re.MULTILINE)

refine2_pattern = "Submitted[ ]by.{0,100}|^เขียนโดย.{0,100}|^Poste?d?[ ]{0,2}(?:by|on){0,2}.{0,100}|^เมื่อวาน[ ]{0,2}\d.{0,100}|^อาทิตย์นี้[ ]{0,2}\d.{0,100}|^อาทิตย์ที่แล้ว[ ]{0,2}\d.{0,100}|^เดือนนี้[ ]{0,2}\d.{0,100}|^เดือนที่แล้ว[ ]{0,2}\d.{0,100}|^รวมผู้เยี่ยมชม[ ]{0,2}\d.{0,100}|^จำนวนผู้ชมโดยประมาณ[ ]{0,2}\d.{0,100}|^รหัสสินค้า[ ]{0,2}\d.{0,100}|^บาร์โค้ด[ ]{0,2}\d.{0,100}|^[ ]โดย[ ]{0,2}.{0,100}|^เข้าชม[ ]{0,2}\d.{0,100}|^โหวต[ ]{0,2}\d.{0,100}|^มุมมอง[ ]{0,2}\d.{0,100}"
refine2_re = re.compile(refine2_pattern, re.MULTILINE)

refine3_pattern = "^[^@\n]{0,30}@\d{0,10}.{0,30}|.{0,100}[-]$|\d*[ ]*x[ ]\d*[^ ][ ]?|^ดูหนัง[ ]?(?:ออนไลน์)?[ ].{0,60}|^คุ้มค่าที่สุดอันดับ[ ]{0,2}\d{0,2}.{0,80}|^เปิด[^\d\n]+.{0,10}"
refine3_re = re.compile(refine3_pattern, re.MULTILINE)

refine4_pattern = "^[^\n]{0,50}คลิก\)|[Ff]acebook[ ]{0,2}(?:\d{0,3},?\d{0,3})[ ]{0,2}เข้าชม|^[ ]{0,2}[^ ]{0,20}[ ]{0,2}\d{0,9}[ ]{0,2}ความเห็น|\[url=.{0,100}|^ผู้ชม[ ]{0,2}(?:\d{0,3},?)+|\([ ]?\)"
refine4_re = re.compile(refine4_pattern, re.MULTILINE)

refine5_pattern = "^[^\d]{0,30}\d{0,10}[ ]{0,2}views.{0,100}|^Prev.{0,100}Next|^สินค้าติดต่อที่.{0,100}|^อ่านต่อคลิก.{0,100}|^สินค้าโปรโมชั่น.{0,200}|^US[ ]?\$\d{0,3},?\d{0,3}.?\d{0,3}.{0,50}"
refine5_re = re.compile(refine5_pattern, re.MULTILINE)

refine6_pattern = "^เจ้าหน้าที่ฝ่ายขาย:\n.{0,80}|^(?:\*+[ ]{0,2}[^\*\n]{0,50}[ ]{0,2}\*+)[ ]{0,2}[\+]?(?:(?:\d{0,3},?)+)?|[\*\+]{2,5}|^(?:[^:\n]{0,30}:).{0,200}"
refine6_re = re.compile(refine6_pattern, re.MULTILINE)

refine7_pattern = "\(?อ่าน[ ]{0,2}\d{0,3},?\d{0,3}[ ]{0,2}(?:ครั้ง[ ]{0,2})?\)?|โพสต์[ ].{0,100}|Read[ ]{0,2}\d{0,9}[ ]{0,2}times|[^ \n]{0,20}[ ]{0,2}pantip|^Previous (?:Post|article).{0,150}|^Next (?:Post|article).{0,150}|^ตอบกลับ[ ]{0,2}.{0,200}"
refine7_re = re.compile(refine7_pattern, re.MULTILINE)

refine8_pattern = "^[ ]?(?:[Pp]ostby|[Pp]osted[ ](?:by|on)).*|^[ ]?เข้าชม/ผู้ติดตาม.*|^[ ]?จำนวนผู้ชมโดยประมาณ[ :]?.*|^[ ]?ลงประกาศฟรี[ ].*|^\|[ ]|^[ ]?จาก[ ].*|^[ ]?By.*|^[ ]{0,2}?โดย[ ]{0,2}?.*"
refine8_re = re.compile(refine8_pattern, re.MULTILINE)

refine9_pattern = "^[^\n\.]{0,60}\.{3}$|^[^\n]{0,30}ฉบับที่[ ].*|^Home[ ]/[ ].{100}|^[^\n\|]{0,60}\|.{0,60}"
refine9_re = re.compile(refine9_pattern, re.MULTILINE)

refine10_pattern = "^[ ]?(?:\)|↑|►|←|«)[ ]?|^[-_]+"
refine10_re = re.compile(refine10_pattern, re.MULTILINE)

refine11_pattern = "^สถิติ(?:วันนี้|สัปดาห์นี้|เมื่อวาน(?:นี้)?|เดือนนี้)[ ]{0,2}.{0,50}|Online[ ]สถิติ.{0,50}"
refine11_re = re.compile(refine11_pattern, re.MULTILINE)

refine12_pattern = (
    "^[^\n\(]{0,80}\(รายละเอียด\)[ ]\(แจ้งลิงก์เสีย\)|^ด[\. ][ชญ][\. ].*|\.{5,}"
)
refine12_re = re.compile(refine12_pattern, re.MULTILINE)

refine13_pattern = "^[ ]?(?:เรื่องย่อ[ ].{0,100}|คุ้มค่าที่สุดอันดับ.{0,100}|คุ้[ ]่าที่สุดอันดับ.{0,100}|\(?ลงโฆษณาฟรี[ ].{0,200}|\(free[ ]online[ ].{0,100}|\(คลิกเพื่อดูต้นฉบับ\)[ ].{0,100}|แก้ไขครั้งสุดท้ายโดย[ ].{0,100})|^[^\d\n]{0,30}[ ]\d{0,3},?\d{0,3}[ ]ครั้ง.{0,50}"
refine13_re = re.compile(refine13_pattern, re.MULTILINE)

refine14_pattern = "^(?:[฿$]?\d{0,9}\.?,?\d{0,9}-?–?:?/?(?:[ ]{0,2}x[ ]{0,2}\d{0,8})?(?:\\bกม\\b\.?)?(?:\\bน\\b\.?)?(?:ล้าน|แสน|หมื่น|พัน|ร้อย|สิบ|บาท|[ ])?){0,5}"
refine14_re = re.compile(refine14_pattern, re.MULTILINE)
