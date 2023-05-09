import re


#### Gamble Clean Words
gamble_words = ['พนัน','แทง','แทงบอล','บาคารา','บา คา รา','เกมพนัน','คาสิโน','คา สิ โน','หวย','สล็อต','กาสิโน','casino','slot','เลขเด็ด','สูตรหวย','a s i n o','sbobet','fun88','ufabet','บาคาร่า','บา คา ร่า','รูเล็ต','ทำนายฝัน','เลขเด่น','สรุปผลบอล','ไฮไลท์ฟุตบอล','วิเคราะห์บอล','ดูบอลสด','พรีเมียร์ลีก','บอลประจำวัน','บอลเต็ง','บอลเด็ด','องค์ลงรวย','สูตรปลาตะเพียน','สามตัวตรง','วิเคราะห์ข้อมูลล่าง','ต่อ ครึ่งลูก','ครึ่งลูกลบ','เสมอควบครึ่ง','ครึ่งควบลูก']

#### Sale Clean Words
sale_skip_words = ['สอบราคา','จัดซื้อจัดจ้าง','ชมรม','สมาคม','นักลงทุน','นักการตลาด','ของกลาง','การลงทุน','นักวิเคราะห์','ขายให้แก่ประชาชน','การลดต้นทุน','การเสนอราคา','กระทรวง','ตลาดหลักทรัพย์','ยอดขายไม่ดี','ยอดขายไม่ค่อยดี','ผู้ประกอบการธุรกิจ','ออกใบอนุญาต','ผู้ประกอบกิจการ']
sale_url_words = ['alibaba.com','shopee.co.th','lazada.com','DocPlayer.net','Alibaba','AliExpress','Aliexpress','TripAdvisor','jobbkk.com']
sale_words = ['ขาย','ซ่อม','ราคา','มือสอง','เช่า','ครีม','ฝ้ากระ','จุดด่างดำ','รับส่วนลด','โปรโมชั่น','กวดวิชา','ติวเตอร์','SEO','คอร์สเรียน SEO','จำหน่าย','ทัวร์','สินค้ามาใหม่','สินค้าทั้งหมด','รีวิวสินค้า','เคสกันกระแทก','ประกาศ','ลงขายของ','เลือกขนาด','บริการจัดส่ง','จัดอันดับ','คาราโอเกะ','จำหน่าย','หาเงินออนไลน์','สั่งซื้อ','ลดกระหนำ่','รหัส','ลงประกาศฟรี','หยิบใส่ตะกร้า','สนใจ','ซื้อ','สินค้า','ผลิตภัณฑ์']

#### Rent Clean Words
skip_words = ['สอบราคา','จัดซื้อจัดจ้าง','ชมรม','สมาคม','นักลงทุน','นักการตลาด','ของกลาง','การลงทุน','นักวิเคราะห์','ขายให้แก่ประชาชน','การลดต้นทุน','การเสนอราคา','กระทรวง','ตลาดหลักทรัพย์']
rent_words = ['บ้านมือสอง','ให้เช่า','เช่า','บ้านเดี่ยว','อพาร์ทเม้นท์','อสังหาริมทรัพย์','เพนท์เฮ้าส์','ทาวน์เฮ้าส์']

#### Script Clean Words
script_words = ['function','var','click','margin','width','height','return','else','alert','<br>','href']

#### Garbage Clean Words
garbage_words = ['โหงวเฮ้ง','ครีมฟอกสี','ครีมผิวขาว','ฟอกสี','ไวท์เทนนิ่งครีม','ครีมไวท์เทนนิ่ง','ครีมลบฝ้ากระ','รับสร้างบ้าน','ครีมโรคสะเก็ดเงิน','บริการจองตั๋ว','บริการรีดผ้า','อาหารเสริมลดน้ำหนัก','ยาลดน้ำหนัก','ลดไขมัน','ผิงโซดา','สร้างบ้าน','ช่างกุญแจ','ช่างโลหะ','ช่างโยธา','ช่างเครื่องยนต์','ช่างไม้','ช่างกลโรงงาน','ช่างไฟฟ้า','ปรสิต','หนอน','เวิร์ม']

#### Football teams
football_teams = ['ยูเวนตุส','อินเตอร์ มิลาน','นาโปลี','เอซี มิลาน','ลาซิโอ','โรม่า','กัลโซ่','เอฟเวอร์ตัน','ซันเดอร์แลนด์','ลิเวอร์พูล','แมนเชสเตอร์','นิวคาสเซิล','เชลซี','อาร์เซนอล','คลิสตัลพาเลช','เซาแทมป์ตัน','เซาแธมป์ตัน','ไฮไลท์ฟุตบอล']

#### Hotels Advertising
hotel_ad = ['โรงแรมอันดับ','ที่พักแบบพิเศษอันดับ','สถานที่พักอันดับ','สถานที่พักคุ้มค่าอันดับ','โรงแรมใกล้กับ','โรงแรมที่ใกล้']

#########
# PRE-COMPILE REGEX to object for speed up processing.
#########
#-----------------------------------------------------
# Remove useless row that make overhead in regex processing

# Unusual row - line size too large
# if there are 3 large lines ( 1500 characters each)
toolarge_line_pattern = '.{1500}'
toolarge_re = re.compile(toolarge_line_pattern, re.MULTILINE)

sidebar_pattern = '.{0,40}(?:(?:\[|\()\d{0,9}(?:\]|\))(?:[ ]{0,2})?,?)'
sidebar_re = re.compile(sidebar_pattern, re.MULTILINE)

nonechar_pattern = '๮|๞|๨|๡|๷|๻|๫'
nonechar_re = re.compile(nonechar_pattern, re.MULTILINE)

none_tone_mark_pattern = 'ก าหนด|เป าหมาย|พ ฒนา|ค ณภาพ|ว จ ย|ค ณล กษณะ|ต างๆ|เป น |ให |บร หาร|ปร บปร ง|ใหม|อย าง|เง น'
none_tone_mark_re = re.compile(none_tone_mark_pattern, re.MULTILINE)

#-----------------------------------------------------

gamble_pattern = '|'.join(gamble_words)
gamble_re = re.compile(gamble_pattern, re.MULTILINE)

football_pattern = '|'.join(football_teams)
football_re = re.compile(football_pattern, re.MULTILINE)

hotel_ad_pattern = '|'.join(hotel_ad)
hotel_ad_re = re.compile(hotel_ad_pattern, re.MULTILINE)

sale_pattern = '|'.join(sale_words)
sale_re = re.compile(sale_pattern, re.MULTILINE)

rent_pattern = '|'.join(rent_words)
rent_re = re.compile(rent_pattern, re.MULTILINE)

json_pattern = "\s*\"(?:\w)*\"\s*:"
json_re = re.compile(json_pattern, re.MULTILINE)

script_pattern = r'\b' + '|'.join(script_words) + r'\b'
script_re = re.compile(script_pattern, re.MULTILINE)

garbage_pattern = '|'.join(garbage_words)
garbage_re = re.compile(garbage_pattern, re.MULTILINE)

ghost_pattern = 'เธฃเน|เธเธญ|เธเน|เธฐเธ|เธฅเธฐ|เธซเธฒ|เธญเธฒ|เธดเธ|เธตเธข|เธญเน|เธญเธ|เธดเน|เธฑเธ|เธกเน|เธฒเธ|เธชเน|เน€เธ'
ghost_re = re.compile(ghost_pattern, re.MULTILINE)

url_pattern = '\\b(?:(?:https?|ftp)://[^\s/$\.\?#].[^\s]*)\\b|\\b(?:www\.?)?(?:(?:[\w-]*)\.)*(?:com|net|org|info|biz|me|io|co|asia|xyz|th|cn|in|uk|jp|ru)\\b'
url_re = re.compile(url_pattern, re.MULTILINE)

menu1_pattern = '\|(?:[^\|\n]*\|)+.*'
menu1_re = re.compile(menu1_pattern, re.MULTILINE)

menu2_pattern = '\|(?:[^\|\n]*\|)+'
menu2_re = re.compile(menu2_pattern, re.MULTILINE)

menu3_pattern = '(?:(?:[^/\n]*/){4,}.*)'
menu3_re = re.compile(menu3_pattern, re.MULTILINE)

menu4_pattern = '[^\n]{0,20}[ ]{0,2}[>»\\\\].*'
menu4_re = re.compile(menu4_pattern, re.MULTILINE)

hashtag_pattern = '#\d*[ ].{0,300}|#(?:(?:[^ \n]*)[ ]?)+|^Tag Archives[ ]{0,2}:.{0,300}|^Posts Tagged[ ]{0,2}:.{0,300}|^HASTAG[ ]{0,2}:.{0,300}|^Tag[s]?[ ]{0,2}:.{0,300}|^Tagged[ ].{0,300}'
hashtag_re = re.compile(hashtag_pattern, re.MULTILINE)

page_pattern = '(?:<<[ ])?(?:ก่อนหน้า|ย้อนกลับ)[ ]{0,2}(?:\[[ ]?\d{0,6}[ ]?\]|[ ]?\d{0,6}[ ]?)*(?:ต่อไป|หน้าถัดไป|ถัดไป)?(?:[ ]?>>)?|<<(?:[ ]\d{0,6}[ ]\-[ ]\d{0,6})+[ ].{0,100}'
page_re = re.compile(page_pattern, re.MULTILINE)

markup_pattern = '\{\{[^\}]*\}\}|\{\{.*'
markup_re = re.compile(markup_pattern, re.MULTILINE)

embeded_java_pattern = '<%[ ]*[^%]*%>|<%.*'
embeded_java_re = re.compile(embeded_java_pattern, re.MULTILINE)

u_pattern = '(?:\uFEFF|\u00AD|[\u200A-\u200F]|\uFFFD|[\uE000-\uF8FF]|[\u202A-\u202C]|\u0092|[\u0091-\u0096]|\u2028|\u2066|\u2069|\u008d|\u0081|\u008E)'
u_re = re.compile(u_pattern, re.MULTILINE)

iframe_pattern = '<iframe.*?<\/iframe>\s*|<iframe.*'
iframe_re = re.compile(iframe_pattern, re.MULTILINE)

block_pattern = '(?:\[[^\]]*\])|(?:«[^»]*»)|(?:<<([^>]*)>>)'
block_re = re.compile(block_pattern, re.MULTILINE)

email_pattern = '(?:(?:([Ee]?mail|อีเมล์)[ ]{0,2}:?[ ]{0,5})?)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
email_re = re.compile(email_pattern, re.MULTILINE)

ip_pattern = '\((?:\d{0,3}\.)+\d{0,3}\)|\(?IP:(?:\d{0,3}\.)+\d{0,3}\)?'
ip_re = re.compile(ip_pattern, re.MULTILINE)

tel_pattern = '(?:(?:[Pp]hone|[Mm]obile|มือถือ|Tel|TEL|Fax|FAX|เบอร์โทรศัพท์|เลขโทรศัพท์|เบอร์ติดต่อ|โทรศัพท์|โทรสาร[ ]{0,2}:|เบอร์โทร|โทร[ ]{0,2}:|โทร\.|โทร[ ]|ติดต่อที่[ ]{0,2}:?|ติดต่อ[ ]{0,2}:?)[ ]{0,2}):?(?:(?:[ ]{0,2})?(?:(?:\d{3}-\d{7})|(?:\d{4}-\d{6})|(?:\d{3}-\d{3}-\d{4}|(?:\d{3}-\d{3}-\d{3})|(?:\d{1}-\d{4}-\d{4})|(?:\d{2}-\d{3}-\d{4})|(?:\d{2}\s\d{3}\s\d{4})|(?:\d{2}-\d{7})|(?:\d{3}\s\d{3}\s\d{4})|(?:\d{3}\s\d{3}\s\d{3})|(?:\d{10})))[ ]{0,2},?)+'
tel_re = re.compile(tel_pattern, re.MULTILINE)

date1_pattern = '(?:(?:การปรับปรุงปัจจุบัน|ตั้งแต่|ลงประกาศเมื่อ|อัพเดทล่าสุด|แก้ไขครั้งสุดท้าย|แก้ไขครั้งล่าสุด|เผยแพร่เมื่อ|เผยแพร่|เขียนเมื่อ|ตอบเมื่อ|เมื่อ|เขียนวันที่|วันที่|วัน)?(?:[ ]{0,2}:[ ]{0,2})?(?:จันทร์|อังคาร|พุธ|พฤหัสบดี|พฤหัสฯ?\.?|ศุกร์|เสาร์|อาทิตย์|จ\.|อ\.|พ\.|พฤ\.|ศ\.|ส\.|อา\.?)?(?:[ ]{0,2}ที่)?(?:[ ]{0,2}[\w\u0E01-\u0E5B]*[ ]{0,2}(?:,|(?:-|\u2013)))?(?:\d{1,4}[ ]{0,2}-)?[ ]{0,2}\d{0,4}(?:-|[ ]{0,2})(?:เดือน[ ]{0,2})?(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.?ค\.?|ก\.?พ\.?|มี\.?ค\.?|เม\.?ย\.?|พ\.?ค\.?|มิ\.?ย\.?|ก\.?ค\.?|ส\.?ค\.?|ก\.?ย\.?|ต\.?ค\.?|พ\.?ย\.?|ธ\.?ค\.?|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|/)(?:-|[ ]|,]){0,2}(?:[ ]{0,2})?(?:\d{4}|\d{2})?,?(?:(?:[ ]{0,2}\d{4}|\d{2}:\d{2}:\d{2})|(?:[ ]{0,2}\d{1,2}:\d{2}(?::\d{2})?[ ]*(?:(?:(?:p|P|a|A)(?:m|M)))?)|[ ]{0,2}เวลา[ ]{0,2}\d{1,2}:\d{2}:\d{2})?(?:[ ]{0,2}น\.)?(?:[ ]{0,2}:[ ]{0,2}\d{1,2}(?:\.|:)\d{2}[ ]{0,2}น\.)?(?:[ ]{0,2}เวลา[ ]{0,2}\d{1,2}:\d{2}[ ]{0,2}[pPaA][mM][ ]{0,2}(?:PDT)?)?(?:[ ]{0,2}เวลา[ ]{0,2}\d{1,2}(?:\.|:)\d{2}[ ]{0,2}(?:น\.)?)?(?:[ ]*\d*[ ]{0,9}ผู้ชม[ ]{0,2}\d)?(?:[ ]{0,2}เวลา[ ]{0,2}:[ ]{0,2}\d{1,2}:\d{2}:\d{2}[ ](?:น\.)?)?(?:[ ]{0,2}เข้าชม[ ]{0,2}\d*[ ]{0,8}ครั้ง)?(?:[ ]{0,2}ที่[ ]{0,2}\d{1,2}:\d{2}[ ]{0,2}(?:[pPaA][mM])?)?(?:[ ]{0,2}Views:[ ]{0,2}(?:\d{0,3},?){0,3})?(?:[ ]{0,2}\(\d{1,2}:\d{2}[ ](?:น\.)?[ ]{0,2}\)(?:[ ]{0,2}ความคิดเห็น[ ]{0,2}\d)?)?(?:[ ]{0,2}-[ ]{0,2}\d{1,2}:\d{2}[ ]{0,2}(?:น\.)?)?(?:[ ]{0,2}จำนวนเข้าชม:(?:[ ]{0,2}\d{0,9},?)*)?(?:[ ]*พ\.ศ\.[ ]{0,2}\d{4}[ ]{0,2}(?:\d{1,2}:\d{2}[ ]{0,2}(?:น\.)?)?)?(?:(?:\d{0,3},?){0,3}[ ]{0,2}ครั้ง)?(?:[ ]{0,2}-\d{1,2}(?:\.|:)?\d{2}[ ]{0,2}(?:น\.)?)?(?:เวลา[ ]{0,2}\d{1,2}:\d{2}:\d{2})?(?:[ ]{0,2}(?:ถึง|จนถึง))?)(?:,[ ]{0,2})?(?:[ ]{0,2}(?:[pPaA][mM])?)?|(?:(?:[ ]{0,2}(?:\d{4}|\d{2})-\d{1,2}-(?:\d{4}|\d{1,2}))(?:[ ]{0,2},[ ]{0,2})?(?:\d{1,2}(?:\.|:)\d{2}[ ]{0,2}(?:#\d*)?(?:น\.)?)?(?:[ ]{0,2}\d{1,2}:\d{2}:\d{2})?)|(?:(?:เปิดบริการ[ ]{0,2})?(?:เวลา[ ]{0,2}\d{1,2}:\d{2}-\d{1,2}:\d{2}[ ]{0,2}(?:น\.)?))|(?:(?:Time[ ]Online[ ]:[ ]{0,2})?(?:\d{1,2}:\d{2}:\d{2})(?:[ ]{0,2}น\.)?(?:[ ]{0,2}[pPaA][mM])?)|(?:\(\d{1,2}:\d{2}[ ]{0,2}-[ ]{0,2}\d{1,2}:\d{2}(?:[ ]*น\.)?(?:[ ]{0,2}[pPaA][mM])?\))|(?:นี้[ ]{0,2}เวลา[ ]{0,2}\d{1,2}(?:\.|:)\d{2}[ ]{0,2}(?:น\.)?)|(?:\d{1,2}[ ]{0,2}(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม))'
date1_re = re.compile(date1_pattern, re.MULTILINE)
date2_pattern = '[พค]\.?ศ\.?[ ]{0,2}\d{4}|\d{4}[ ]{0,2}เวลา[ ]{0,2}\d{2}:?\.?\d{2}(?:[ ][Pp][Mm])|[พค]\.?ศ\.'
date2_re = re.compile(date2_pattern, re.MULTILINE)

html_pattern = '<br>?|&nbsp|{\s*document\..*|SELECT.*FROM.*WHERE.*|<a\s*href=.*|<img\s*src=.*'
html_re = re.compile(html_pattern, re.MULTILINE)


final1_pattern = '^(?:[฿$]?\d{0,9}\.?,?\d{0,9}-?–?:?/?(?:[ ]{0,2}x[ ]{0,2}\d{0,8})?(?:\\bกม\\b\.?)?(?:\\bน\\b\.?)?(?:ล้าน|แสน|หมื่น|พัน|ร้อย|สิบ|บาท|[ ])?){0,5}'
final1_re = re.compile(final1_pattern, re.MULTILINE)

final2_pattern = '^[ ]?ตอนที่[ ]*\d{0,3}(?:[-–]?\d{0,3})?[ ]{0,2}.{0,100}|^สั่ง.{0,50}บาท|^[ ]?เลขจดแจ้ง[ ]{0,2}.{0,13}|^.{0,100}\.jpg[ ]{0,2}\(.{0,50}|^.{0,20}รายการ|^[ ]?สนใจ[ ]{0,2}.{0,15}โทร[ ]{0,2}.{0,12}|^[ ]?ผู้แสดงความคิดเห็น.{0,60}|^\(.{0,40}[ ]{0,2}\d{0,5}[ ]{0,2}.{0,10}\).{0,200}|^[ ]?ผู้เข้าชมทั้งหมด.{0,30}|^[ ]?ฉบับที่[ ]{0,2}\d{0,7}[^-–]{0,30}-?–?[ ]|^[ ]?โพสต์ที่แชร์โดย.{0,200}|^[ ]?Copyright.{0,200}|กำลังแสดงหน้าที.{0,200}|[ ]{0,2}รีวิว.{0,100}|^[ ]?ข้อที่ \d{0,4}|^เข้าชม/ผู้ติดตาม.{0,13}'
final2_re = re.compile(final2_pattern, re.MULTILINE)

final3_pattern = 'Submitted[ ]by.{0,100}|^เขียนโดย.{0,100}|^Poste?d?[ ]{0,2}(?:by|on){0,2}.{0,100}|^เมื่อวาน[ ]{0,2}\d.{0,100}|^อาทิตย์นี้[ ]{0,2}\d.{0,100}|^อาทิตย์ที่แล้ว[ ]{0,2}\d.{0,100}|^เดือนนี้[ ]{0,2}\d.{0,100}|^เดือนที่แล้ว[ ]{0,2}\d.{0,100}|^รวมผู้เยี่ยมชม[ ]{0,2}\d.{0,100}|^จำนวนผู้ชมโดยประมาณ[ ]{0,2}\d.{0,100}|^รหัสสินค้า[ ]{0,2}\d.{0,100}|^บาร์โค้ด[ ]{0,2}\d.{0,100}|^[ ]โดย[ ]{0,2}.{0,100}|^เข้าชม[ ]{0,2}\d.{0,100}|^โหวต[ ]{0,2}\d.{0,100}|^มุมมอง[ ]{0,2}\d.{0,100}'
final3_re = re.compile(final3_pattern, re.MULTILINE)

final4_pattern = '^[^@\n]{0,30}@\d{0,10}.{0,30}|.{0,100}[-]$|\d*[ ]*x[ ]\d*[^ ][ ]?|^ดูหนัง[ ]?(?:ออนไลน์)?[ ].{0,60}|^คุ้มค่าที่สุดอันดับ[ ]{0,2}\d{0,2}.{0,80}|^เปิด[^\d\n]+.{0,10}'
final4_re = re.compile(final4_pattern, re.MULTILINE)

final5_pattern = '^[^\n]{0,50}คลิก\)|[Ff]acebook[ ]{0,2}(?:\d{0,3},?\d{0,3})[ ]{0,2}เข้าชม|^[ ]{0,2}[^ ]{0,20}[ ]{0,2}\d{0,9}[ ]{0,2}ความเห็น|\[url=.{0,100}|^ผู้ชม[ ]{0,2}(?:\d{0,3},?)+|\([ ]?\)'
final5_re = re.compile(final5_pattern, re.MULTILINE)

final6_pattern = '^[^\d]{0,30}\d{0,10}[ ]{0,2}views.{0,100}|^Prev.{0,100}Next|^สินค้าติดต่อที่.{0,100}|^อ่านต่อคลิก.{0,100}|^สินค้าโปรโมชั่น.{0,200}|^US[ ]?\$\d{0,3},?\d{0,3}.?\d{0,3}.{0,50}'
final6_re = re.compile(final6_pattern, re.MULTILINE)

final7_pattern = '^เจ้าหน้าที่ฝ่ายขาย:\n.{0,80}|^(?:\*+[ ]{0,2}[^\*\n]{0,50}[ ]{0,2}\*+)[ ]{0,2}[\+]?(?:(?:\d{0,3},?)+)?|[\*\+]{2,5}|^(?:[^:\n]{0,30}:).{0,100}'
final7_re = re.compile(final7_pattern, re.MULTILINE)

final8_pattern = '\(?อ่าน[ ]{0,2}\d{0,3},?\d{0,3}[ ]{0,2}(?:ครั้ง[ ]{0,2})?\)?|โพสต์[ ].{0,100}|Read[ ]{0,2}\d{0,9}[ ]{0,2}times|[^ \n]{0,20}[ ]{0,2}pantip|^Previous (?:Post|article).{0,150}|^Next (?:Post|article).{0,150}|^ตอบกลับ[ ]{0,2}.{0,200}'
final8_re = re.compile(final8_pattern, re.MULTILINE)

final9_pattern = '^[ ]?(?:[Pp]ostby|[Pp]osted[ ](?:by|on)).*|^[ ]?เข้าชม/ผู้ติดตาม.*|^[ ]?จำนวนผู้ชมโดยประมาณ[ :]?.*|^[ ]?ลงประกาศฟรี[ ].*|^\|[ ]|^[ ]?จาก[ ].*|^[ ]?By.*'
final9_re = re.compile(final9_pattern, re.MULTILINE)

final10_pattern = '^[^\n\.]{0,60}\.{3}$|^[^\n]{0,30}ฉบับที่[ ].*|^Home[ ]/[ ].{100}|^[^\n\|]{0,60}\|.{0,60}'
final10_re = re.compile(final10_pattern, re.MULTILINE)

# ---- IP address
final11_pattern = '\(?(?:\d{0,3}\.)+\d{0,3}\)?|\(?IP[: ](?:\d{0,3}\.)+\d{0,3}\)?'
final11_re = re.compile(final11_pattern, re.MULTILINE)

final12_pattern = '^[ ]?(?:\)|↑|►|←|«)[ ]?'
final12_re = re.compile(final12_pattern, re.MULTILINE)

final13_pattern = '^สถิติ(?:วันนี้|สัปดาห์นี้|เมื่อวาน(?:นี้)?|เดือนนี้)[ ]{0,2}.{0,50}|Online[ ]สถิติ.{0,50}'
final13_re = re.compile(final13_pattern, re.MULTILINE)

final14_pattern = '^[^\n\(]{0,80}\(รายละเอียด\)[ ]\(แจ้งลิงก์เสีย\)'
final14_re = re.compile(final14_pattern, re.MULTILINE)

final15_pattern = '^[ ]?(?:เรื่องย่อ[ ].{0,100}|คุ้มค่าที่สุดอันดับ.{0,100}|คุ้[ ]่าที่สุดอันดับ.{0,100}|\(?ลงโฆษณาฟรี[ ].{0,200}|\(free[ ]online[ ].{0,100}|\(คลิกเพื่อดูต้นฉบับ\)[ ].{0,100}|แก้ไขครั้งสุดท้ายโดย[ ].{0,100})|^[^\d\n]{0,30}[ ]\d{0,3},?\d{0,3}[ ]ครั้ง.{0,50}'
final15_re = re.compile(final15_pattern, re.MULTILINE)

# ---- Hex Characters
final16_pattern = '(?<![A-Za-z0-9\-ก-๙])(?:[0-9A-Fa-f]{2,})(?![0-9A-Za-z\-ก-๙])'
final16_re = re.compile(final16_pattern, re.MULTILINE)


def clean_text(text: str) -> str:


    found = False

    # ---- Clean too large unused lines 
    # Limit matches list to 2 items only, enough
    matches = toolarge_re.findall(text)[:2]
    # Classify as toolarge row if number of matches >= 2
    if len(matches) >= 2:
        return ""

    # ---- Clean none characters row
    # Limit matches list to 30 items 
    matches = nonechar_re.findall(text)[:30]
    # Classify as none character row if number of matches >= 30
    if len(matches) >= 30:
        return ""
        
    # ---- Clean none tone mark row
    # Limit matches list to 30 items 
    matches = none_tone_mark_re.findall(text)[:30]
    # Classify as none tone mark row if number of matches >= 30
    if len(matches) >= 30:
        return ""

    # ---- Clean Gamble ~ 9.2% of mC4 data
    # if found gamble word 2 times in a row, classify as gamble row 
    # remove the row
    # Limit matches list to 2 items only, enough
    matches = gamble_re.findall(text)[:2]
    # Classify as gamble if number of matches >= 2
    if len(matches) >= 2:
        return ""

    # ---- Clean Football data 
    # if found gamble word 4 times in a row, classify as football data
    # remove the row
    # Limit matches list to 4 items only
    matches = football_re.findall(text)[:4]
    if len(matches) >= 4:
        return ""

    # ---- Clean Hotel Advertising
    # if found word 4 times in a row, classify as Hotel Ad. data
    # remove the row
    # Limit matches list to 4 items only, enough
    matches = hotel_ad_re.findall(text)[:4]
    if len(matches) >= 4:
        return ""
    
    # ----  Clean Sale ~26% of mC4 data
    # Sale row data is diverse, 
    # so the regex is not used in this case.
    # Rules:
    # 1. Remove row if it contains common specific Sale's URL
    # 2. Skip this row if it contains specific keywords, eg. "สอบราคา", "จัดซื้อจัดจ้าง, etc."
    # 3. Scan the row with sale keywords, if there are at leat 3 sale kewords found then remove the row.
    
    # url
    for w in sale_url_words:
        ix = text.find(w)
        if ix != -1:
            return ""
    
    # skip specific sale data (not clean out)
    skip = False
    for w in sale_skip_words:
        ix = text.find(w)
        if ix != -1:
            skip = True
            break

    if not skip:
        # Classify as Sale data ( matches >= 3, can be adjusted)
        matches = sale_re.findall(text)[:3]
        if len(matches) >= 3:
            return ""


    # ---- Clean Rent (พวกเช่า ~2% of mC4 data)
    # Rent use another rules
    # 1. find skip words in the row. If found, get next row (not remove)
    # 2. if found rent word 2 times in a row, classify as rent row 
    #    remove the row

    # skip 
    for w in skip_words:
        ix = text.find(w)
        if ix != -1:
            return ""

    # Limit matches list to 2 items only, enough
    matches = rent_re.findall(text)[:2]
    # Classify as rent if number of matches >= 2
    if len(matches) >= 2:
        return ""


    # ---- Clean pattern (json like -> "abc": ~.5-1% )
    # 99% can classify as gabage: so remove them
    # match n items to make sure they are garbages n=20, can change
    matches = json_re.findall(text)[:20]
    # if match only 20+, classify as garbage
    if len(matches) >= 20:
        return ""


    # ---- Clean script (Javascript, etc. ~.5% )
    # 99% can classify as gabage: so remove them
    matches = script_re.findall(text)[:10]
    # Classify as script if number of matches > 10
    if len(matches) >= 10:
        return ""


    # ---- Clean garbage (useless or not necessary ~.45%)
    # classify as gabage: so remove them
    matches = garbage_re.findall(text)[:4]
    # Classify as garbage if number of matches >= 4
    if len(matches) >= 4:
        return ""


    # ---- Clean ghost language (~0.008% can cancel this clean) 
    # classify as ghost : so remove them
    matches = ghost_re.findall(text)[:4]
    # Classify as ghost if number of matches >= 4
    if len(matches) >= 4:
        return ""


# --------------------------------------------------------------- 
# The row that falls down here is 
# the row that passed all romove filters.
# Now, we will scan and REPLACE unwanted characters and patterns
# with ' ' (blank)  
# --------------------------------------------------------------- 

# -- พวก URL
    """
text = '''http://www.dobrolubie.ru/forum/index.php?showuser=59153
http://www.uzsat.net/index.php?showuser=22989  ตัวหนังสือแทรก
http://www.prevedrussia.ru/projectrussiaclub/forum/index.php?showuser=17747
http://www.feedagg.com/feed/583/member.php?u=215917
มีข้อความข้างหน้า  http://www.bizmama.ru/forum/index.php?showuser=35150  มีข้อความต่อ
http://www.jagfile.info/forum/index.php?action=profile;u=6233
http://forum.tvoi.net.ua/index.php?showuser=32548
http://bellydance-sakh.ru/forum/index.php?showuser=71723
http://www.thainationalfilm.com/site/webboard/profile.php?mode=viewprofile&u=91394
http://www.forum.lacrimosafan.ru/index.php?showuser=24795
http://www.akiba-online.com/forum/member.php?u=509353
http://forum.streetbox.ru/index.php?s=0e393c8c0bffcf96760525b9fcfa164c&showuser=41952      มีข้อความต่อไป
http://www.hizkod.com/member.php?7-cacomeza
http://jedeni.ru/index.php?showuser=5431
http://forum.sa.volgocity.ru/index.php?showuser=369628
https://iapi.bot.or.th/Stat/Stat-ExchangeRate/DAILY_AVG_EXG_RATE_V1/?start_period=2018-02-01&end_period=2018-02-01&currency=USD
http://radiofreewashington.com http://cankardeslernakliyat.com http://survivalsolarkit.com http://nosscholengemeenschap.com http://viajesferrer.com http://pennsylvania-resorts.comrow=862
http://www.journal-social.mcu.ac.th/wp-content/uploads/2015/06/%E0%B9%90%E0%B9%97-%E0%B8%94%E0%B8%A7%E0%B8%87%E0%B9%83%E0%B8%88.pdf
https://example.com/search?query=language+model&page=2#results
alibaba.com  shopee.co.th   look.mp4  data.pdf
ห้อง.เดี่ยว  ปลาทอง.jp
'''
    """
    text = url_re.sub(' ',text)


# -- พวก Menu pattern '|' (1)
    """
text = '''| 18/05/61 | เปิดดู 2285 | หมวด คอนโดมิเนียม | กรุงเทพมหานคร | ดูแผนที่
| 18/05/61 | เปิดดู 3125 | หมวด คอนโดมิเนียม | กรุงเทพมหานคร | ดูแผนที่
| 18/05/61 | เปิดดู 1974 | หมวด คอนโดมิเนียม | ชลบุรี | ดูแผนที่
| 16/03/61 | เปิดดู 2170 | หมวด คอนโดมิเนียม | กรุงเทพมหานคร | ดูแผนที่
| 13/11/60 | เปิดดู 2022 | หมวด คอนโดมิเนียม | กรุงเทพมหานคร | ดูแผนที่
| 16/06/60 | เปิดดู 1861 | หมวด คอนโดมิเนียม ผู้สูงอายุ | กรุงเทพมหานคร | ดูแผนที่
|เริ่มต้น  abcde efgh | ijklm
ข้อความทดสอบ|ab/d|de fg
    เริ่มข้อความ |20/05/62| 22/05/63 | สิ้นสุดข้อความ'''
    """
    text = menu1_re.sub(' ',text)


# -- พวก Menu pattern '|' อีกแบบ (2)
    """
text = '''อาไวเกมส์เยือนแพ้รวด ทีเด็ดบอลวางฟลูมิเนนเซเบาๆ
    |23 ก.ย 2562| 22 ก.ย 2562| 21 ก.ย 2562| 20 ก.ย 2562| 19 ก.ย 2562| 18 ก.ย 2562| 17 ก.ย 2562| 16 ก.ย 2562| 15 ก.ย 2562| 14 ก.ย 2562| 13 ก.ย 2562| 12 ก.ย 2562| 11 ก.ย 2562| 10 ก.ย 2562| 9 ก.ย 2562| 8 ก.ย 2562| 7 ก.ย 2562| 6 ก.ย 2562| 5 ก.ย 2562| 4 ก.ย 2562| 3 ก.ย 2562|

|abcde|ddffadf kkljlj| abcdef '''
    """
    text = menu2_re.sub(' ',text)


# -- พวก Menu pattern '/' (3)
    """
text = '''คำค้น หมวดหมู่ ทุกหมวดหมู่ - APPLE - iPhone X - iPhone 8 Plus / 7 Plus - iPhone 8 / 7 - iPhone 6 Plus / 6S Plus - iPhone 6 / 6S - iPhone 5 / 5S / SE- SAMSUNG - Galaxy Note FE / Note 7 - Galaxy J7+ - Galaxy Note 8 - Galaxy J7 PRO (2017) / J730 - Galaxy J5 PRO (2017) / J530 - Galaxy S8 Plus - Galaxy S8 - Galaxy A7 (2017) - Galaxy A5 (2017) - Galaxy A3 (2017) - Galaxy J5 Prime - Galaxy J7 Prime - Galaxy S7 Edge - Galaxy S7 - Galaxy Note 5 - Galaxy S6 Edge - Galaxy S6- HUAWEI - Mate 10 PRO - Mate 10 - Nova 2i - P10 Plus - P10 - Mate 9 PRO - Mate 9 - GR5 2017 / Honor 6X - P9 Plus - P9 - P9 Lite - Mate 8 - Nova Plus - Y6II - Y5II - P8 - P8 Lite - Mate 7 - MediaPad M3 8.4 - MediaPad M2 10.0 - MediaPad M2 8.0- LG - G6- GOOGLE - Pixel 2 XL - Pixel 2- ONEPLUS (1+) - 1+5 - 1+3 / 3T- XIAOMI - Mi Max 2 - Mi 6 - Mi 5S Plu
เพื่อทดสอบ

คำค้น หมวดหมู่ ทุกหมวดหมู่ - APPLE - iPhone X - iPhone 8 Plus / 7 Plus - iPhone 8 / 7 - iPhone 6 Plus / 6S Plus - iPhone 6 / 6S - iPhone 5 / 5S

(1) 400 กรัม/น้ำหนึ่งลิตร  (2) 300 กม./ชม.  (3) 50บาท/กก.  (4) อัตราส่วน 3/8  <- ต้องมี 4 ชุดขึ้นไปใน 1 line ถึงจะนับว่าเป็นเมนู'''
    """
    text = menu3_re.sub(' ',text)


# -- พวก Menu pattern '>','»','\' -  (4)
    """
text = '''บ้าน > ผลิตภัณฑ์ > ผงเคลือบ Ral9006 
--- เริ่ม 
บ้าน > ฟิมล์ > กล้อง > เลนส์
หลังห้องดอทคอม » เคล็ดลับสู่ความสำเร็จ » 5 วิธีเสพติดความสำเร็จที่ยิ่งใหญ่
Home \ อุตสาหกรรม \ โรงงานผลิตเคมีภัณฑ์ \ บริษัท ยูไนเต็ด ซิลิกา (สยาม) จำกัด
--- สิ้นสุด่
'''
    """
    text = menu4_re.sub(' ',text)


# -- พวก Hashtag
    """
text = '''เมกเกอร์หญิงรุ่นเล็กขอแสดงฝีมือ        3,947 views     More +#จัดปาร์ตี้สละโสด #แต่งงาน #มาร์กี้ #ดาราแต่งงาน #นางเอง #มาร์กี้แต่งงาน #ว่าที่สะไภ้หมื่นล้าน #มิ้นคิมร่วมปาร์ตี้ #ราศี
#มาร์กี้แต่งงาน #ดาราแต่งงาน #มาร์กี้ #แต่งงาน #ฉลองวันครบรอบ 
หลังจากที่ประกาศแต่งงานว่าจะแต่งกันในช่วงปลายปี ในเดือนธันวาคมที่จะถึงนี้ ซึ้งทั้งคู่มีเวลาเปลี่ยมตัวและเตรียมการ ไม่มากนัก
#มากี้บินดูชุดแต่งงาน #ข่าวดารา #ราศี #ชุดแต่งงานมาร์กี้ #มาร์กี้แต่งงาน #ชุดแต่งงานดารา
ความลับถูกเปิด!!
#หลวงพ่อหวั่น วัดคลองคูณ #รวยม…
#หลวงพ่อหวั่น วัดคลองคูณ #รวยมหาเศรษฐี พร้อมส่งค่ะชื่อชุดดิน : สมุทรปราการ Samut Prakan (Sm)

Tag Archives: ทิปวิธีการลดหุ่น
Posts Tagged: "ทิปวิธีการลดหุ่น"ขณะนี้ วันที่ : 14 Google (66.249.71.60) วันนี้ ด้วยดีไซน์ของตัวเครื่องที่เพรียวบาง พกพาสะดวก หน้าจอ Retina ประสิทธิภาพที่ยอดเยี่ยมสำหรับการทำงานประจำวัน แบ

HASTAG : เทียนกง เจนนิเฟอร์ ลอว์เรนซ์ เตรียมทำสงครามในตัวอย่างแรกของ The Hunger Games: Mockingjay – Part 1 – JEDIYUTH

Tags: A lot, english, most, People, Some, คน, ตัวอย่าง, ประโยค, ประโยคถูก, ประโยคผิด, ผิด, ผิดบ่อย, ผิดพลาด, ผู้คน, ภาษา, ภาษาอังกฤษ, ศาสนา, สัญชาติ, สับสน, อังกฤษ, เชื้อชาติ, แกรมม่า, แก้ไข, ไวยากรณ์

Tag: แยกสุขุมวิท

ขอบคุณบทความจาก : Tags : ยูพิค,สมัคร Youpik,Youpk คอมมิชชั่น
#3850 yulk (@piliying) (จากตอนที่ 13)พี่มาร์คปากแข็งจีจีแตฉันชอบแกอ่า
#3645 JUNLINEZ (@pipo_gummy) (จากตอนที่ 13)อ่านแล้วคะป้าครึ่งตอนแรกชอบมากเลย ความรักที่ทั้งสองมีให้กันแบบรักกันมากๆ รอนะคะ
Tagged ฝรั่งเศส,เก้าอี้,Elementcommun
'''
    """
    text = hashtag_re.sub(' ',text)


# -- พวก Pagination
    """
พบ 1,023 ตำแหน่ง << 1 - 20 21 - 40 41 - 60 61 - 80 81 - 100 101 - 120 121 - 140 141 - 160 161 - 180 181 - 200 201 - 220 221 - 240 241 - 260 261 - 280 281 - 300 301 - 320 321 - 340 341 - 360 361 - 380 381 - 400 401 - 420 421 - 440 441 - 460 461 - 480 481 - 500 501 - 520 521 - 540 541 - 560 561 - 580 581 - 600 601 - 620 621 - 640 641 - 660 661 - 680 681 - 700 701 - 720 721 - 740 741 - 760 761 - 780 781 - 800 801 - 820 821 - 840 841 - 860 861 - 880 881 - 900 901 - 920 921 - 940 941 - 960 961 - 980 981 - 1000 100 
งานสิทธิบัตรและประกันสังคม
ข่าวประกวดราคาจัดซื้อ/จัดจ้างทั้งหมด 921 รายการ : 47 หน้า : << ย้อนกลับ [ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ][ 8 ][ 9 ][ 10 ][ 11 ] 12 [ 13 ][ 14 ][ 15 ][ 16 ][ 17 ][ 18 ][ 19 ][ 20 ][ 21 ][ 22 ][ 23 ][ 24 ][ 25 ][ 26 ][ 27 ][ 28 ][ 29 ][ 30 ][ 31 ][ 32 ][ 33 ][ 34 ][ 35 ][ 36 ][ 37 ][ 38 ][ 39 ][ 40 ][ 41 ][ 42 ][ 43 ][ 44 ][ 45 ][ 46 ][ 47 ] หน้าถัดไป>>
text= '''<< ก่อนหน้า 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 1147 1148 1149 1150 1151 1152 1153 1154 1155 1156 1157 1158 1159 1160 1161 1162 1163 1164 1165 1166 1167 1168 1169 1170 1171 1172 1173 1174 1175 1176 1177 1178 1179 1180 ถัดไป>>วันที่ 18 กรกฎาคม 2562 ชม 99564 ครั้ง'''
text = '''ก่อนหน้า 1 2 3 4 [5] 6 7 8 9 10 11 12 ต่อไป<U+FEFF><U+FEFF> โครงการบ้านศรุตาปา	"""
    text = page_re.sub(' ',text)


# -- พวก Pattern หมวดหมู่ (เช่นพวก sidebar ของ Wordpress)
    """
text = '''วันเกิด: dolceCliny (31), okzizNeeby (42), GlaydsiMop (30), biolaBapsSaups (32), tertuSer (41), KlemenVap (33), Michaelerymn (30), DouglasZek (33), matbewotly (32), entnived (40), maslinAL (29), sivatVAL (29), alminTIM (28), sumPAVV (29), iganYUR (29), dejinHycle (37), ssurvgoabbedia (32), laveyKnino (32), Robertorifs (42), nisDENN (29), Bradleyfluow (41), MorganTox (33), dismIVAN (29), semNIKK (29), gisVALL (29), LarryNat (38), felmetix (39)
วันเกิด: DanielLam (40), sunliPhank (32), rioriJiP (35), Miguelnom (31), nikitendug (34), matmureend (42), FrancuaCiz (42), DavidordEd (33), countoxync (32), RigelTal (40), ThomasVed (38), spinpVaf (39), EtelJoips (33), TerryPrupt (41), Edahagind (33), Aroztum (42), remtekes (40), GeorgeDit (31), WileyDup (33), ForeSuich (32), Berkezes (43), Angersuity (30), Mylorek (36), ShawnToF (39), Larryceafe (37), RezanNus (39), rencoloasy (40), lamodBip (35), talogrob (33), EugeneWen (38), Igorjus (43), Michaelimicy (37)  kljlj;jljljljk (25),kjkljlj(37) lsjfljljljl'''
text = '''สินค้าทั้งหมด [194]
ลำโพง/Speaker [173]ลำโพง Audioengine [11]ลำโพง Audioengine A2+ Powered Speaker [4]ลำโพง Audioengine B2 Bluetooth Speaker [3]ลำโพง Audioengine A5+ Powered Speaker [4]ลำโพง B&O Bang & Olufsen [3]ลำโพงพกพา B&O BeoPlay A2 [3]ลำโพง Creative [2]ลำโพง Creative SoundBlaster Roar [1]ลำโพง SOUND BLASTER ROAR PRO [1]ลำโพงพกพา Dreamwave [2]Dreamwave TREMOR [1]ลำโพง Dreamwave Explorer [1]ลำโพงพกพา Divoom [2]ลำโพง Divoom Onbeat 500 Gen2 [2]ลำโพง Dope [2]ลำโพงพกพา Dope Duo link [1]ลำโพง Dope Billionaire [1]ลำโพง EDIFIER [10]ลำโพง Edifier R1700BT [1] abcdefg hijklmnopq$
'''
    """
    text = sidebar_re.sub(' ',text)


# -- พวก Markup Language (Django,Jinja2,Liquid template, etc.)
    """
text = '''{{ kv.owner.preferred_name | truncate:25 }}{{ kv.owner.fullname | truncate:40 }} ผู้ติดตาม: {{ kv.owner.follower_count | number }} ติดตาม: {{ kv.owner.followee_count | number }} ติดต่อ ติดตาม เลิกติดตาม {{ kv.owner.preferred_name | truncate:25 }}{{ kv.owner.fullname | truncate:40 }}
เห็นชื่ออีเมลส่งอีเมลแจ้งด้วยเมื่อรายการนี้มีความเห็นเพิ่มเติม ใส่รูปหรือไฟล์คลิกใส่รูปหรือไฟล์ใหม่คลิกเพื่อใส่ไฟล์ที่มีอยู่แล้ว {{ file.asset_file_name โบนัสฟรี 500 _เล่นพนัน ภาษาอังกฤษ_เว็บบอลแจกเครดิตฟรี
Toronto, ON M3B2X5
#B107 -1396 DON MILLS RD , Toronto, ON M3B2X5
To: #B107 -1396 Don Mills Rd, Toronto, ON'''
    """
    text = markup_re.sub(' ',text)


# -- พวก Embedded Javascript ( like Node.js, Express.js, etc.)
    """
text = '''<% if ( total_view > 0 ) { %> <%= total_view > 1 ? "total views" : "total view" %>, <% if ( today_view > 0 ) { %> <%= today_view > 1 ? "views today" : "view today" %> no views today       No views yet <% uncompleted format'''
    """
    text = embeded_java_re.sub(' ',text)


# -- พวก <U+....>
    """
text = '''ต่อไป<U+FEFF><U+FEFF> โครงการบ้านศรุตาปา'''
    """
    text = u_re.sub(' ',text)

# -- พวก iframe
    """
text = '''<iframe src='https://vod.thaipbs.or.th/videos/Qnm8mDJnNHJf/embedded?title=%E0%B8%97%E0%B8%B1%E0%B9%88%E0%B8%A7%E0%B8%96%E0%B8%B4%E0%B9%88%E0%B8%99%E0%B9%81%E0%B8%94%E0%B8%99%E0%B9%84%E0%B8%97%E0%B8%A2+-+%E0%B8%AD%E0%B8%B4%E0%B9%88%E0%B8%A1%E0%B8%AA%E0%B8%B8%E0%B8%82%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%9A%E0%B9%89%E0%B8%B2%E0%B8%99%E0%B8%9A%E0%B8%B2%E0%B8%87%E0%B9%80%E0%B8%9A%E0%B9%89%E0%B8%B2+%E0%B8%AB%E0%B8%B8%E0%B8%9A%E0%B9%80%E0%B8%82%E0%B8%B2%E0%B8%A5%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%97%E0%B8%B0%E0%B9%80%E0%B8%A5+%E0%B8%88.%E0%B8%95%E0%B8%A3%E0%B8%B2%E0%B8%94&image=https%3A%2F%2Fthaipbs-program.s3-ap-southeast-1.amazonaws.com%2Fcontent%2Fimages%2Fepisode%2F1%2FF1%2F2F%2F1F12F0bpOUtP-large.jpg&tags=%7B%22video_category%22%3A%22show%22%2C%22program%22%3A%22tuathin%22%2C%22category%22%3A%22lifestyle%22%7D' border='0' width='640px;' height='360px;' allowfullscreen allow='autoplay; fullscreen' class='watch-embedded-player'></iframe> มีข้อความต่อเนื่อง  ที่มีประโยชน์   <iframe kljlj </iframe>       นี่ก็อีกข้อความหนึ่งที่มีประโยชน์ เช่นกัน    <iframe abcdef ghijk</iframe>klkjkljklj    <iframe hello world   </iframe>  มีข้อมูลต่อ  <iframe width="560" height="315" src=" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen allowtransparency><  เจาะลึกโทรศัพท์มือถือสำหรับเล่นเกม (Gaming Mobile) ที่มาพร้อมสเปกแรงจัดเต็ม โดยความเป็นจริงแล้วแตกต่   <iframe      
เจาะลึกโทรศัพท์มือถือสำหรับเล่นเกม (Gaming Mobile) ที่มาพร้อมสเปกแรงจัดเต็ม โดยความเป็นจริงแล้วแตกต่    <iframe อากาศร้อนมาก </iframe>  ดื่มน้ำมากๆนะ
'''
    """
    text = iframe_re.sub(' ',text)

# -- พวก [...] , <<...>> และ «...»
    """
text = 'สวัสดีครับ [abcdefghi] วันนี้อากาศร้อน [asdfgsgsdg] ระวังโรคลมแดด [abcdefg][...]« ตอบ ข้อความที่อยู่ระหว่าง » เดี๋ยวไม่สบาย << ก็มันร้อนจริงๆ  >>'
    """
    text = block_re.sub(' ',text)

    # -- พวก Email
    """
text = '''Email : r_faisu@hotmail.comเครื่องบดสมุนไพร เพื่อทำยาป้องกันโรคต่างๆ - เครื่องบดยาสมุนไพร คุณภาพเยี่ยมราคาถูก ได้มาตรฐานระดับสากล ISO จัดส่งฟรีทั่วประเทศ : Inspired by LnwShop.com
เครื่องบดสมุนไพร เพื่อทำยาป้อง…
มาที่อีเมล์ zonedara@gmail.com ข่าวข
mail : nongluck_joy@hotmail.com  email i17bkk@gmail.com  ok ใช้ได้ดี'''
    """
    text = email_re.sub(' ',text)


#--- Tel
    """
text = '''
เบอร์โทรศัพท์ : 063-9547533
เบอร์โทรศัพท์ : 063-9547533  สำนักงาน เบอร์โทรศัพท์ : 063-9547533  Fax : 063-9998888
เลขโทรศัพท์ 081-842-5373
TEL : 0832-741706              4500   9,000 บาท
โทร: 045-612-611
Fax: 045-613-088
โทรสาร : 0-2370-2700         089623333  ประเทศไทย  0123456789
เบอร์ติดต่อ 02 185 2865
เบอร์ติดต่อ 02-305-6652 , 084-4523432
โทร. 077 975 522, 098 016 6775
โทรศัพท์ : 084 469 8604, 091 756 7763
โทรศัพท์ : 081 810 5006, 081 822 3026, 087 070 9595
โทรศัพท์ : 081 810 5006, 081 822 3026, 087 070 9595 , 0832-741706, 0899395848, 084-4523432
โทรศัพท์ 02-681-8390
    องค์การโทรศัพท์ 
เบอร์โทร : 0833485952
เบอร์โทร:085-3228934
0899395848
mobile: 084-544-1234      hellomobile:
phone : 02-2581122
มือถือ: 084-554-4444            เอามือถือ ไปวาง
ติดต่อ: 02-2605549      เดินทางติดต่อ
'''
    """
    text = tel_re.sub(' ',text)

# --- พวกวันที่ รูปแบบหลากหลาย
    """
โดย Teetawat » พุธ 19 มิ.ย. 2019 11:51 am
โดย Teetawat » พฤหัสฯ. 20 มิ.ย. 2019 3:49 pm
Saber, 28 กรกฎาคม 2018
แก้ไขครั้งล่าสุด: 28 กรกฎาคม 2018
purinankun, 28 กรกฎาคม 2018
purinankun, 28 กรกฎาคม2018
เริ่มประมูล พฤ. 11 ม.ค 2561 14:09:06
ปิดประมูล พฤ. 25 ม.ค 2561 14:09:06ชาวสีม่วง ! 
วันที่ 29 ก.ค. 59 เวลา 13:22:18 น.
วันที่ 29 กค. 59 เวลา 13:22:18 น.
วันที่ 29 กค 59 เวลา 13:22:18 น.
วันที่ 29 กค59 เวลา 13:22:18 น.
วันที่ 4-5 สิงหาคม เวลา 10.30-13.30 น. ณเดชน์-ญาญ่า 
จ.ขอนแก่น วันที่ 22 ตุลาคม
19 พฤศจิกายน 2018 : 12.29 น.
29-04-2010, 20:41 #3
2023/04/15
15/04/2023
15/04/2023
29-04-2012
เมื่อ ก.ค. 26, 2017 เวลา 8:59pm PDT
เมื่อ ก.ค. 19, 2017 เวลา 8:53am PDT
เขียนวันที่ 27 มกราคม 2020 24 มกราคม 2020 โดย admin
เผยแพร่เมื่อ: 23 ธันวาคม 2558
tapinki - 26 ก.ค. 12    1,455
Time Online : 24:16:59
[url= ปัญหาของเครื่องBB5กับอีมี่123456แต่Providerยังอยู่[ Google เข้าเยี่ยมชมหน้านี้ล่าสุดเมื่อ : 26 มกราคม 2563, 10:04:19 Powered by SMF © 2006–2020, Simple Machines LLC เรือนกระจกอ:
ตอบเมื่อ: 26 ส.ค. 2006, 9:03 pm
การปรับปรุงปัจจุบัน: 02-พฤษภาคม-2554 ถึง 15-สิงหาคม-2554 ดี ถึง
ตั้งแต่ ตุลาคม 2554 จนถึง กุมภาพันธ์ 2556
วันอังคารที่ ๑๘ มิถุนายน พ.ศ. ๒๕๖๒ ๐๔:๕๔ น. © 2007 - 2019 Tasang Limited, All Rights Reserved 0.0426s วันศุกร์ 22 กุมภาพันธ์ พ.ศ.2562 36551084 ครั้ง นับตั้งแต่ เสาร์ 02 กุมภาพันธ์, 2008คู่มือสอบเจ้าพนักงาน
ราศีกุมภ์ 9 - 15 ธันวาคม 2562
ดวงรายสัปดาห์ ราศีกุมภ์ 9 - 15 ธันวาคม 2562
ราศีกุมภ์ 9 – 15 ธันวาคม 2562
ราศีกุมภ์ 9 - 15 ธันวาคม 2562
อัพเดท : 07-08-55, 14:48 น.
อัพเดทล่าสุด : 27 Dec 2018 11:32:41 น.
เริ่มประมูล พฤ. 11 ม.ค 2561 14:09:06
ปิดประมูล พฤ. 25 ม.ค 2561 14:09:06ชาวสีม่วง
เผยแพร่เมื่อ: 09 ธันวาคม 2558 วันที่ 5 ธันวาคม 2558 คณะครูและบุคลากร
คลิปตัวอย่างหนัง Cilp วันที่ 26 Jul 2012 เวลา 17:38
ความคิดเห็นที่ 8 พุธ ที่ 16 เดือน มกราคม พ.ศ.2562 เวลา 02:29:17
ลงประกาศเมื่อ 18 ต.ค. 2559 อัพเดทล่าสุด 18 ต.ค. 2559 15:19:02 น. เข้าชม 214 ครั้ง
ลงประกาศเมื่อ 20 พ.ค. 2562 อัพเดทล่าสุด 20 พ.ค. 2562 11:40:48 น. เข้าชม 57 ครั้ง
Marvel เผยแพร่: 04 เม.ย. 2019 จำนวนเข้าชม: 10,280
ผู้ลงประกาศ : นางาสาว แพรพลอย ตะลีงอกะลี
อัพเดทล่าสุด : 18 ต.ค. 2559 15:19:02 น. เข้าชม 15 ครั้ง
แก้ไขครั้งล่าสุด: 28 กรกฎาคม 2018
เผยแพร่เมื่อ: 23 ธันวาคม 2558
เมื่อ 18 ก.ค. 62 14:14:34
วันที่ 29 ก.ค. 59 เวลา 13:22:18 น.
วันที่: 21 กุมภาพันธ์ 2560 เวลา:17:30:00 น.
19 พฤศจิกายน 2018 : 11.03 น.
29-04-2010, 20:41 #3
30-04-2010, 10:27 #4
30-04-2010, 10:42 #5
เสาร์ที่ 1 มิถุนายน 2562 00:00:53 น.
23 ก.พ. 53 เวลา 10.07 น. 5131 ผู้ชม 0
23 ธันวาคม 2558
กรกฎาคม 28, 2014 ที่ 7:37 PM
กรกฎาคม 29, 2014 ที่ 12:39 PM
กรกฎาคม 29, 2014 ที่ 6:37 PM
28 December 2012 - 14:41 น.
28-5-54
พ.ศ.
    พ.ศ. ๒๔๘๐
พ.ศ. 2480   พศ 2480
สารวัด ฉบับที่ 151222 ค.ศ. 2019 สัปดา 
เนื้อหาถูกสร้างขึ้นเมื่อ th, 2019 เวลา 11:01 pm อ
03 ส.ค. 53 (06:30 น.) ความคิดเห็น 3
2018-10-2 9:56:28
เขียนเมื่อ 12 กรกฎาคม 2558
เปิดบริการ เวลา 17:00-01:00 น.
วันที่ 19 เมษายน 2562 - 19:39 น.
12 พ.ย. 2019 Views: 58,124
08 พ.ย. 2019 Views: 2,901
05 พ.ย. 2019 Views: 94,476
03 ก.ย. 2019 Views: 1,945
Post By : autoflight P591506169288
Posted by driftworm , ผู้อ่าน : 1958 , 12:32:29 น.
งปธน.จีน_china.com
2019-06-24 15:02:59 Xinhua News Agency
ข่าวในจีนApril 08 2020 21:30:36ตั้งตี้ ชวนก๊วนขาแดนซ์ มาโดด Touch 2nd Anniversary Jumper Dance Party 12 มีนาคมนี้ 2009 ข่าวเกมส์ไทย , เกมส์บนเว็บ 3 ปี ago gamededteam 0
ใกล้ถึงเวลาแห่งความสนุกกันแล้วซิ กับปาร์ตี้โดด เด้ง แดนซ์ ใน “Touch 2nd Anniversary Jumper Dance Party” ที่จะเกิดขึ้น ในวันเสาร์ที 12 มีนาคมนี้ เวลา 12.00 น.เป็นต้นไป ณ Bounce Thailand โดยเพื่อนๆขาแดนซ์ จะได้พบกับความสนุกมากมายที่ทีมงานคัดสรรมาเอาใจกันโดยเฉพาะเลย ทั้งภารกิจสุดโหด แรร์ไอเทมพิเศษ และพบปะพูดคุยกันในแมทช์พิเศษกับเหล่า MC อย่างใกล้ชิด สนิทใจ งานนี้รับรองคุ้มถูกใจชาวเกมเมอร์แน่นอน!!
– หนุ่มสาวชาวทัชออนไลน์ที่จะเข้าร่วมงาน สามารถคลิกลงทะเบียนได้ที่นี้เลย – โดยหนุ่มสาวชาวทัชออนไลน์ที่ใช้ไอดีเฟสบุ๊คสำหรับเข้าเล่นเกม กรุณาเตรียมจดไอดีของท่านมาให้พร้อมสำหรับขั้นตอนการลงทะเบียนหน้างานในวันที่ 12 มีนาคม 2559 ด้วยนะคะ
----  พวกเวลาราชการ เก็บไว้ให้ให้ LLM เรียน เวลาเปิดปิด ครับ 
- คลินิกในเวลาราชการ วันจันทร์ – วันศุกร์ เวลา 07.00 –16.00 น.
- คลินิกนอกเวลาราชการ วันจันทร์ – วันศุกร์ เวลา 16.00 – 20.00 น. วันเสาร์ เวลา 07.00 – 14.00 น.
ในเวลาราชการ วันจันทร์-วันศุกร์ เวลา 08.00 – 16.00 น.
นอกเวลาราชการ วันจันทร์-วันศุกร์ เวลา 16.00 – 20.00 น.
นอกเวลาราชการ วันเสาร์-วันอาทิตย์ เวลา 09.00 – 12.00 น.
· ในเวลาราชการ เวลา 7.00 – 16.00 น.
· นอกเวลาราชการ วันเสาร์ เวลา 7.00 – 16.00 น.
· วันอาทิตย์และวันหยุดนักขัตฤกษ์ เวลา 7.00 - 12.00 น.
วันจันทร์ – วันศุกร์ เวลา 8.00 –12.00 น.
วันจันทร์ – วันศุกร์ เวลา 13.00 – 16.00 น.
วันศุกร์ (เว้นวันพุธ) 9.00 – 16.00น.
'''
    """
    text = date1_re.sub(' ',text)
    text = date2_re.sub(' ',text)


#— html, script, sql  — try to sweep up leftover again .09% + .04% = 0.13%
    """
text = '''
<br
1.&nbspขนมปังฝรั่งเศสหั่นชิ้น&nbsp100&nbspกรัม
2.&nbspกล้วยหอม&nbsp1&nbspลูก
5"เพจจีน" แฉ !! ร้านซีฟู้ดภูเก็ตโกง ตร.จับดำเนินคดี { document.cookie = "refresh_token=;expires= GMT;domain=.trueid.net;path= กิ๊กส์หน้าแหก! แดเนี่ยล เจมส์ ไม่ใช่ผู้เล่นแมนฯ ยูไนเต็ดที่เร็วที่สุดในฤดูกาลนี้ หลังกิ๊กส์บอกว่าเจมส์วิ่งเร็วที่สุดเท่าที่เขาเคยเห็นมา
ซุปดอกกะหล่ำ&nbspCauliflower&nbspSoup
เพิ่มผลงานใหม่)
<br ธันวา บุญสูงเนิน (ชื่อเล่น: ทอย; เกิด ) ชื่อในวงการเพลง The Toys เป็นนักร้อง นักดนตรี นักแต่งเพลงและ ]ชาว[[ประเทศไทย ดึงข้อมูลจาก " ข้อมูลประกอบพร้อมวิธีการเลือกใช้บริการผู้รับทำ seo – bloggnytt.org
<a href=" title="เกมส์Chickaboom"><img src=" alt="เกมส์Chickaboom"
<img src=

SELECT `AttachFile`.`id`, `AttachFile`.`lesson_id`, `AttachFile`.`title`, `AttachFile`.`description`, `AttachFile`.`created`, `AttachFile`.`modified` FROM `ln_learnsquare`.`ln_attach_files` AS `AttachFile` WHERE `AttachFile`.`lesson_id` IN (636, 637, 638, 639, 939, 940, 941, 2066, 2067, 2068, 2069, 2404, 2405, 2406, 6119, 6120, 6121) 0 0 0
'''
    """
    text = html_re.sub(' ',text)


#--- final refinement (พวกเก็บกวาดสุดท้าย)
# Methods:
# 1. replace patterns with blanks.
# 2. found specific pattern remove that line.
# Patterns:
# 1. Start with number and only numbers in the line.
# 2. Start with words + ' ' + NUMBER.
# 3. Statistic information and common web/social statistic patterns.
# 4. Start with single meaning word/words only in a line. 
# 5. Start with characters + ":" + any/none characters.
# 6. Start with some common/popular social media keywords.
# 7. Contains common/pupular keywords
# 8. Pagination keywords ( not specify in previous filter, can be rewrite and put to pagination filter above. 
# ... etc.
    """
text = '''2.00 72.1
5.00 180.2
10.00 360.5
20.00 720.9
50.00 1802.3
100.00 3604.5
200.00 7209.0
500.00 18,022.6
500.0 13.87
1000.0 27.74
2000.0 55.49
5000.0 138.71
10,000.0 277.43
20,000.0 554.86
50,000.0 1387.15

10,000 0.046
20,000 0.092
50,000 0.229
100,000 0.458
200,000 0.917
500,000 2.291
1,000,000 4.583
2,000,000 9.166
5,000,000 22.914
10,000,000 45.829
1,000,000,000 4582.891
2,000,000,000 9165.781
0.050 10,900
0.100 21,825
0.200 43,650
0.500 109,100
1.000 218,200
2.000 436,400
5.000 1,091,025
เปิดเพจ 444,501
60 x 640 2880 x 2560 1080 x 960 1024 x 1024 1024 x 768 1080 x 1920 1152 x 864 1200 x 1024 1280 x 1024 1280 x 768 1280 x 800 1280 x 854 1280 x 960 1440 x 1280 1440 x 900 1600 x 1200 1600 x 900 1680 x 1050 1920 x 1200 2048 x 1536 2048 x 2048 2160 x 1920 400 x 800 480 x 800 540 x 960 600 x 800 640 x 1136 640 x 480 640 x 960 720 x 1280 720 x 720 750 x 1334 768 x 1024 800 x 1003 800 x 1280 800 x 480 800 x 600 800 x 853 960 x 544 960 x 800
บันเทิง-
ผู้ชม 502 ผู้ชม
( )
เปิด
รีวิว Locanda Paolo
[url= กุญแจหาย[  นักเรียนแชมป์โลกโครงงานวิทยาศาสตร์ - wanidaarawan
JEDIYUTH 6 ความเห็น
สรุปผลบอลเมื่อคืนนี้ รายละเอียดด้านใน กำลังแสดงหน้าที่ 1 1 2 3 4 5 6 7 8 9  ปัญญา สติ สมาธิ_resize_resize.jpg [ 43.75 
Facebook 85,230 เข้าชม
_______________________________________________________________________________
เฌอชม pantip
@0941543221 Twitter Stats Overview
คุยกันที่ @0941543221  ดีไหมครับ
เข้าชม/ผู้ติดตาม1404.9%
527,472 สมาชิกที่ใช้งาน
เลขจดแจ้ง 040345
10345 
สั่ง 2 ปุก เหลือ ปุกละ 160 บาท
สั่ง 3 ปุก เหลือ ปุกละ 140 บาท
ตอนที่ 1-1 นักล้วงกระเป๋าผู้หลักแหลม 4.5k 1
ตอนที่ 2 นักล้วงกระเป๋าผู้หลักแหลม 1.5k 0
คุ้มค่าที่สุดอันดับ 6 จาก 61 ใน โรงแรมบรรยากาศเงียบสงบในพอร์ตแลนด์
คุ้มค่าที่สุดอันดับ 7 จาก 61 ใน โรงแรมบรรยากาศเงียบสงบในพอร์ตแลนด์
ดูหนัง Hellraiser Bloodline HD
ดูหนัง งาบแล้วไม่งุ่นง่าน 2
ดูหนังออนไลน์ Hellraiser Bloodline
ดูหนังออนไลน์ Hellraiser Bloodline HD
ดูหนังออนไลน์ งาบแล้วไม่งุ่นง่าน 2
image.jpg (106.13 KB, 650x485 - ดู 17 ครั้ง.)
My world ranking : 43,861
12v-24v 4 รายการ
฿0.00 - ฿99.99 5 รายการ
สนใจ หรือโทร 095-951-3663
0:07 – เคล็ดลับต่างๆเกี่ยวกับแตงโม
(พบสินค้า 23 ชิ้น)
ผู้แสดงความคิดเห็น gowell (pui-dot-9536-at-gmail-dot-com)วันที่ตอบ 
ฉบับที่ 151221 ค.ศ. 2019 - วัดพระชนนีของพระเป็นเจ้า รังสิต (โบสถ์คาทอลิก) วัดพระชนนีของพระเป็นเจ้า รังสิต (โบสถ์คาทอลิก)
โพสต์ที่แชร์โดย gggubgib36 (@gggubgib36) *See you at Paragon Hall ❤️ คุณหญิงกีรติพร้อมแล้วววววเจอกัน 15.00 ที่ รอยัลพารากอนฮอล น๊าาาาาาาาาา #มากันเยอะๆ#เจ๊เปาบางพลี . . .#ggbbpp #งดฝากร้านเปา1ขวบ
ผู้เข้าชมทั้งหมด 2,009,396 ครั้ง 2807 557

กำลังแสดงหน้าที่ 1 1 2 3 4 5 6 7 8 9 ปัญญา สติ สมาธิ_resize_resize.jpg [ 43.75 KiB กระบวนการทำงานของมรรค 8_100.1 Kb_resize.jpg [ 45.99 KiB อริยสัจ 4_resize.jpg [ 58.29 KiB 
รีวิวจาก Booking.com (157)2. อาหารเด็ก..ข้าวบดวิตามิน-ไฟเบอร์สูง ...
Copyright 2020 \ Healthy Lifestyle - กีฬายิมนาสติกและฟิตเนส \ วิธีเผาผลาญ 600 แคลอรี่ในหนึ่งชั่วโมงของการออกกำลังกายตุ๊กตา กำลังหา แฟน
(ผู้ดูแล: ₪๐StemCell๐₪) 
ข้อที่ 89
2,474,234 เข้าชม
(คลิกเพื่อดูต้นฉบับ) สงขลาว~2.JPG (32.89 kB, 368x256 - ดู 3271 ครั้ง.)
หน้า: 2 3 4 ... 299
(อ่าน 79 ครั้ง)
Read 4733 times
TrueNews 5999 views     4 likes Shareวัฒนธรรมการเรียนรู้ 
Re: 9 ความเชื่อเรื่องนาฬิกาผ้ายางปูพื้นรถยนต์ Benz E220 W211 # 5926523
Prev1. . .288 289 290 291 292 293 294 295 296 297 . . .3593 Next
38:10 แล้วกษัตริย์มีรับสั่งให้เอเบดเมเลคคนเอธิโอเปียว่า “จงเอาคนไปจากที่นี่กับเจ้าสามสิบคน แล้วฉุดเยเรมีย์ผู้พยากรณ์ออกมาจากคุกใต้ดินก่อนเขาตาย”
1311 โพสต์ • หน้า 131 จากทั้งหมด 132 • 1 ... 128, 129, 130, 131, 132
อ่านต่อคลิก[2 เม.ย. 2558](อ่าน 4,821 ครั้ง)-ไม่มีผลโหวต-
ตอบกลับ ↓        Anonymous กุมภาพันธ์ 8, 2012 ที่ 5:07 am     ถ้าโชคดีก็รอด ถ้าโชคร้ายก็ตายยกครัวถ้าคิดจะทิ้งก็ลองทำซักครั้ง ก็ดีเหมือนกันนะ
ตอบกลับ ↓        Anonymous ตุลาคม 18, 2012 ที่ 9:16 pm      Acer Aspire 4720g เปิดติดแล้วครับ ต้องขอบคุณวิธีที่แนะนำน่ะครับ ^_^
ตอบกลับ ↓        Anonymous มิถุนายน 25, 2013 ที่ 7:59 pm     ลองอยู่ครับเดวมาแจ้งผล
4ล้าน 1607.71ล้าน 4,340
1.97หมื่น 2.25ล้าน 48
7120 3.09แสน 56
99999 999 ล้านบาท
081-4555
ตัวเลขไม่ได้เริ่มที่ต้นแถว  3200 4500 ไม่ตัด
จะใช้ท้ายสุดของการกรองเพื่อเก็บกวาดเท่านั้น ไม่ใช้ก็ได้ 
asfjlkj
7.  caseforsell.com (รายละเอียด) (แจ้งลิงก์เสีย)
8. caseiphone.net (รายละเอียด) (แจ้งลิงก์เสีย)
10.00 - 13.30
10.00 – 10.30 น. พัก
10.30 – 12.00 น. Personalized medicine in action
3434 - 999 น.
** กระดาษดี สีขาว ** 23,345
**ลุงสนามข้างวัดพระแก้ว**+57,664,123,343,3999
***  วัดสระเกศ ***  34
** kjk ** 
+++ iSpeed Shop - ค้ำโช๊ค เกจ์ เบาะ กรองอากาศ ท่อซิลิโคน อื่นๆ +++ (อ่าน 424864 ครั้ง)ไผ่ ส่งเพลงใหม่ "ไปฮักกันสา" แอบส่องนางเอกเอ็มวีหน้าคุ้นๆ
** กระดาษดี สีขาว ** 23,345,234   อ่าน 2200
ประกาศเมื่อ : 03 พ.ค. 61 11:30 น.
เลขบัญชี : 512-20898-53 ชื่อ : วณิชโรจน์ พจน์ทรจรัส
เลขบัญชี : 859-228478-6 ชื่อ : วณิชโรจน์ พจน์ทรจรัส
เลขบัญชี : 54110948046 ชื่อ : วณิชโรจน์ พจน์ทรจรัส
โดย Ponkberry
โดย Darya Wong
โดย starenergyi13 3 749
หัวข้อ : แนวข้อสอบพนักงานมหาวิทยาลัย ราชภัฏ
เขียนโดย Thailand Company บริษัทในหมวดอุตสาหกรรม
เข้าชม: 1096 เชฟ: Ti
เข้าชม 3137 ผ้าทำจากผ้า cotton อย่างดี รูปทรงสวยงาม เหมาะสำหรับใส่ออกกำลังกาย และใส่ยามพักผ่อน
เข้าชม: 8253 เชฟ: Pu
เข้าชม: 7503 เชฟ: Pu
4ล้าน 1607.71ล้าน 4,340
1.97หมื่น 2.25ล้าน 48
7120 3.09แสน 56
มุมมอง 29 702
มุมมอง 902 779
มุมมอง 10,771,538
จำนวนคนอ่าน: 13920 บทความบทสวดมนต์ : คำกล่าวแสดงตนเป็นพุทธมามกะ
Postby snasui » Sun  pm
Posted by Nate Phanwiroj
Posted by ลูกบัว , ผู้อ่าน : 5965 , 
Posted on 5, 20 5, 2018 by mintcss
Posted by thitimedia , ผู้อ่าน : 2949 , 
จำนวนผู้ชม : 2416234
จำนวนผู้โหวต : 1418
โหวต 1418 คน
เพื่อน46 วันที่ : 29 ขอบคุณค่ะที่เตือนสติ
arexy13 วันที่ : 11 เห็นด้วยกับคำว่าลงมือทำครับ
เมื่อวาน 600
อาทิตย์นี้ 1250
อาทิตย์ที่แล้ว 1849
เดือนนี้ 5755
เดือนที่แล้ว 12003
รวมผู้เยี่ยมชม 660583
ต้นไทร หญิง อายุ 61 ปี 5 เดือน (52,8 คลิก)
ต้นไทร หญิง อายุ 61 ปี 5 เดือน 
จำนวนผู้ชมโดยประมาณ฿ 327.75
ความคิดเห็น: 3,770
คะแนนสะสม: 10583 แต้ม
ความคิดเห็น: 64,963
คะแนนสะสม: 948 แต้มพร้อมสำหรับการสูญเสียน้ำหนัก - ลดน้ำหนักใน Siofor 1000
อย่างเช่น Location ฺ: Bangrak ค่ะ
เจ้าหน้าที่ฝ่ายขาย:
คุณเอม 084-107-3797 55 09.00 น. ถึง 18.00 น.
US $10,00.95 - 14.79 / ชิ้น จัดส่งฟรี
0.0 กม. จาก Parkfields Gallery
23.2 กม. จาก Parkfields Gallery
(ลงประกาศฟรี ฉะเชิงเทรา) - กีฬา » ฟุตบอล 15 ก.พ. 2563
สินค้าโปรโมชั่น*ไม่มีส่วนลดใดๆแล้ว*	ตู้เสื้อผ้า ขนาด ก1200xล550xส1800 มม. สีสัก ตู้เสื้อผ้า-บานเปิด PSP โปรโมชั่นวันนี้ ถึง 12 เมษายน 2560BS-101(สีบีช)(สินค้าโปรโมชั่นราคาพิเศษส่งฟรีรอรอบผ่าน)฿2,300
แองเจิลโทร925
Rating: 5 Value: ★★★★★ Reviewed By: Digitaltv Thaitvอยากเดินบนสายงานในการทำเว็บไซต์ควรศึกษาอะไรบ้าง
แหล่งที่มา : refun.com,manager.co.th
ภาพ : www.pinterest.com, www.termsuk.com
Submitted by thep on Re: เลือก Debian mirror ใกล้คุณด้วย http.debian.net
Submitted by Thaitop_DC (not verified)
Twitter: @FreeYOUTHth
Instagram: freeyouth.ig
เขียนโดย Thailand Company บริษัทในหมวดอุตสาหกรรม
รหัสสินค้า 148082
บาร์โค้ด 8852758162309เที่ยวกลางคืน ร้านกินดื่มเน้นค็อกเทลท็อปฮิตในเมืองกรุง คัดมาเน้น ๆ เพื่อสายชิลโดยเฉพาะ – My Blog
Previous PostPrevious การเรียกร้องของนักวิทยาศาสตร์ชาวจีน
Next PostNext Dems จะไม่สนใจผู้มีสิทธิเลือกตั้งในการดูแลสุขภาพอีกหรือ?หัวข้อประกาศ : องค์การเภสัชกรรม (อภ.) เดินหน้าพัฒนายาฟาวิพิราเวียร์ ต้านโควิด-19
Previous Previous post: Japan Only : 4 สิ่งที่ Apple ใส่มาพิเศษใน iPhone 7 ญี่ปุ่น
Next Next post: สรุปใน 1 รูป : กล้อง iPhone 7 Plus เหนือกว่า iPhone 7 !เนื้อเพลง
ที่นอนบุฟองน้ำ 90ซม....
โลชั่นน้ำนมข้าวสุวรร...
กระเป๋าผ้าแฮนด์เมดน่...
ข้าวกล้องเพาะงอกเคลื...
) ทำให้ผู้เรียนได้ประสบการณ์ตรง
) ทำให้ผู้เรียนเข้าใจง่ายและจดจำเรื่องที่สาธิตได้นาน
) ทำให้ผู้เรียนรู้วิธีการแก้ปัญหาได้ด้วยตนเอง
) ทำให้ประหยัดเงินและประหยัดเวลา
ประเภท การศึกษา | หมวด สถาบันสอนวิทยาศา...
ประเภท การศึกษา | หมวด สอนศิลปะ
ประเภท อสังหาริมทรั ์ | หมวด ธุรกิจรับสร้างบ้าน
ประเภท การศึกษา | หมวด สอนศิลปะ
↑ พระราชทานยศนายตำรวจภูธรแลนายตำรวจพระนครบาล
↑ พระราชทานยศ (หน้า ๒๔๗๓)
↑ เรื่อง ย้ายและบรรจุตำแหน่งผู้บังคับการตำรวจ
►การสอบเทียบเครื่องแก้ววัดปริมาตร
►เทคนิคพื้นฐานสำหรับนักจุลชีววิทยา
►อาหารเลี้ยงเชื้อและปฏิกิริยาทางชีวเคมี สำหรับตรวจสอบเชื้อจุลินทรีย์ที่ก่อให้เกิดโรคอาหารเป็นพิษ
← 8มหา’ลัยเอกชน‘ไร้คุณภาพ’ สกอ.ขีดเส้น1เดือนปรับหลักสูตร
← Burning out – อมิตา ทาทา ยัง (TATA YOUNG)
« แม่ค้าไม่ยอมรับว่าขายขนมจีบค้างคืน
Online สถิติทั้งหมด 91,646 คน
สถิติวันนี้ 103 คน สถิติเมื่อวาน 365 คน
สถิติสัปดาห์นี้ 1,240 คน สถิติเดือนนี้ 721 คน
สถิติเมื่อวาน 365 คน
สถิติเดือนนี้ 721 คน
Online สถิติทั้งหมด 91,646 คน
สถิติวันนี้ 103 คน สถิติเมื่อวาน 365 คน
สถิติสัปดาห์นี้ 1,240 คน สถิติเดือนนี้ 721 คน
สถิติเมื่อวาน 365 คน
สถิติเดือนนี้ 721 คน
(รายละเอียด) (แจ้งลิงก์เสีย)
CaseMonster (รายละเอียด) (แจ้งลิงก์เสีย)
เรื่องย่อ Hellraiser Bloodline
คุ้ ่าที่สุดอันดับ 1 จาก 61 ใน โรงแรมบรรยากาศเงียบสงบในพอร์ตแลนด์
(ลงโฆษณาฟรี กรุงเทพมหานค 
(free online classifieds Pakist 
(คลิกเพื่อดูต้นฉบับ) สงขลาว~2.JPG (32.89 kB, 368x256 - ดู 3271 ครั้ง.)
แก้ไขครั้งสุดท้ายโดย รถเต่าเมืองแป้ : เมื่อ 13:42
ได้รับอนุโมทนา 554,750 ครั้ง ใน 7,607 โพสต์
ลิงเจี๊ยก ก 117.47.15.231
(IP:111.84.78.112) ความเห็นเพิ่มเติมที่ 149 (15:50) ยากว่ะทําไม่ได้ยากๆ
'''
    """
    text = final1_re.sub(' ',text)
    text = final2_re.sub(' ',text)
    text = final3_re.sub(' ',text)
    text = final4_re.sub(' ',text)
    text = final5_re.sub(' ',text)
    text = final6_re.sub(' ',text)
    text = final7_re.sub(' ',text)
    text = final8_re.sub(' ',text)
    text = final9_re.sub(' ',text)
    text = final10_re.sub(' ',text)
    text = final11_re.sub(' ',text)
    text = final12_re.sub(' ',text)
    text = final13_re.sub(' ',text)
    text = final14_re.sub(' ',text)
    text = final15_re.sub(' ',text)

    """
#---- Hex characters
E0B89F Tesla-E0 B9 89 E0 miniB8 B2 ระบบเครื่องยนต์E0 B9 E0-version
Angry Bird E0 B9 80 E0 B8 81 E0 B8 A1 E0 B8 AA E0 B9 8C E0 B8 82 E0 B8 B1 E0 B8 9A E0 B8 A3 E0 B8 96 E0 B8 8A E0 B8 99 E0 B8 9C E0 B8 B5 E0 B8 8B E0 B8 AD E0 B8 A1 E0 B8 9A E0 B9 80 E0 B8 81 E0 B8 A1 E0 B8 AA E0 B9 8C E0 B8 97 E0 B8 B3 E0 B9 80 E0 B8 84 E0 B9 89 E0 B8 81 E0 B8 81 E0 B8 A5 E0 B9 89 E0 B8 A7 E0 B8 A2 E0 B8 AB E0 B8 AD% E0 B9 80 E0 B8 81 E0 B8 A1 E0 B9 81 E0 B8 95 E0 B9 88 E0 B8 87 E0 B8 95 E0 B8 B1 E0 B8 A7 E0 B8 99 E0 B8 B2 E0 B8 87 E0 B8 9F E0 B9 89 E0 B8 B2 E0 B9 81 E0 B8 AA E0 B8 99 E0 B8 AA E0 B8 A7 E0 B8 A2 
(D8d6047fe50d166b3f755429c6d0cfbc รวมผลิตภัณฑ์สำหรับ a Core 8 0 สำหรับ E46)
    """
    text = final16_re.sub(' ',text)

    # Clean short lines
    # ( len(line) <= 30 characters , cut this line off)
    lines = text.split("\n")
    tmptext = ''
    for line in lines:
        if len(line) > 30:
            tmptext += line + '\n' 

    text = tmptext

    # ---- The scan row that passes all filter is written to disk
    # before write to disk, get rid of spaces by change them to single space (' ').

    text = re.sub("[' ']+",' ',text, 0, re.MULTILINE)
    text = re.sub("\n\s*",'\n',text, 0, re.MULTILINE)

    return text


def clean_dataset(dataset: List[str]) -> List[str]:
    """
    Description : Call function clean_text to process the whole dataset.
    Input text : An input dataset having each element as a document in the dataset.
    Output : A clean dataset.
    """

    for i, data_point in enumerate(dataset):
        dataset[i] = clean_text(data_point)

    return [data_point for data_point in dataset if data_point != ""]