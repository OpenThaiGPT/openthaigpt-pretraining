# flake8: noqa


def sorted_by_len(keyword_list):
    len_keywords = [[len(kw), kw] for kw in keyword_list]

    return [kw for _, kw in sorted(len_keywords)][::-1]


PORN_KEYWORDS = [
    "คลิปหลุด",
    "กระเจี๊ยว",
    "คลิปโป๊",
    "หนังโป๊",
    "หนังโป้",
    "หนังโป็",
    "เรื่องเสียว",
    "ซอยหี",
    "ชักว่าว",
    "ท่าหมา",
    "ขย่มควย",
    "เล่นเสียว",
    "ควยใหญ่",
    "หนังเอ็กซ์",
    "แหกหี",
    "เย็ด",
]
GAMBLE_KEYWORDS = [
    "ufabet",
    "UFABET",
    "ล้มโต๊ะ",
    "เซียนสเต็ป",
    "บอลเต็ง",
    "แทงบอล",
    "คาสิโน",
    "บาคาร่า",
    "เว็บสล็อต",
    "เกมสล็อต",
    "สล็อตออนไลน์",
    "คาสิโนออนไลน์",
    "หวยมาเลย์",
    "หวยฮานอย",
    "น้ำเต้าปูปลา",
    "หวยออนไลน์",
    "แทงหวย",
    "หวยหุ้น",
    "ยิงปลาออนไลน์",
    "ได้เงินจริง",
    "บา คา ร่า",
]
SPAM_MOVIE_KEYWORDS = [
    "ดูหนังออนไลน์",
    "หนังออนไลน์",
    "เว็บดูหนัง",
    "หนังชนโรง",
    "หนังใหม่ชนโรง",
    "เสียงไทย",
    "เสียงญี่ปุ่น",
    "เสียงอังกฤษ",
]
SPAM_LIKE_KEYWORDS = [
    "ปั้มไลค์",
    "รับจ้างกดไลค์",
    "จ้างไลค์",
    "ปั๊มไลค์",
    "ปั่นไลค์",
    "เพิ่มไลค์",
    "ซื้อไลค์",
]
CODE_KEYWORDS = [
    "padding:",
    "display:",
    "S3=n8",
    "phpBB Debug",
    "getElementById",
    "innerHTML",
    "parseInt",
    "addEventListener",
    "console\.log",
    "checkCookieForTarget",
    "setAttribute",
    "getItem",
    "if \(",
    "else {",
    "JSON\.stringify",
    "onclick",
]

WEBBOARD_KEYWORDS = [
    "คุณกำลังใช้งานแบบปิดการใช้ Javascript",
    "Longdo Dictionary",
    "นโยบายการคุ้มครองข้อมูลส่วนบุคคล",
    "เงื่อนไขการให้บริการเว็บไซต์",
    "นโยบายความปลอดภัย",
    "นโยบายเว็บไซต์และการปฏิเสธความรับผิด",
    "คุณอาจจะยังไม่ได้เข้าสู่ระบบหรือยังไม่ได้ลงทะเบียน",
    "คุณไม่ได้เข้าสู่ระบบหรือคุณไม่มีสิทธิ์เข้าหน้านี้",
]

PORN_KEYWORDS += [" ".join(list(kw)) for kw in PORN_KEYWORDS]
GAMBLE_KEYWORDS += [" ".join(list(kw)) for kw in GAMBLE_KEYWORDS]
SPAM_MOVIE_KEYWORDS += [" ".join(list(kw)) for kw in SPAM_MOVIE_KEYWORDS]

DOCUMENT_REMOVAL_KEYWORDS = (
    PORN_KEYWORDS
    + GAMBLE_KEYWORDS
    + SPAM_MOVIE_KEYWORDS
    + SPAM_LIKE_KEYWORDS
    + CODE_KEYWORDS
    + WEBBOARD_KEYWORDS
)

PARTIAL_REMOVAL_KEYWORDS = [
    "Posted on",
    "Posted by",
    "Posted by:",
    "Posted By:",
    "สมาชิกหมายเลข [0-9,]+",
    "อ่าน [0-9,]+ ครั้ง",
    "เปิดดู [0-9,]+ ครั้ง",
    "ดู [0-9,]+ ครั้ง",
    "คะแนนสะสม: [0-9,]+ แต้ม",
    "ความคิดเห็น: [0-9,]+",
    "[0-9,]+ บุคคลทั่วไป กำลังดูบอร์ดนี้",
    "หน้าที่แล้ว ต่อไป",
    "ความคิดเห็นที่ [0-9,]+",
    "[0-9,]+ สมาชิก และ [0-9,]+ บุคคลทั่วไป",
    "กำลังดูหัวข้อนี้",
    "เข้าสู่ระบบด้วยชื่อผู้ใช้",
    "แสดงกระทู้จาก:",
    "กระทู้: [0-9,]+",
    "เว็บไซต์เรามีการใช้คุกกี้และเก็บข้อมูลผู้ใช้งาน โปรดศึกษาและยอมรับ นโยบายคุ้มครองข้อมูลส่วนบุคคล ก่อนใช้งาน",
    "Privacy & Cookies: This site uses cookies. By continuing to use this website, you agree to their use\.",
    "Previous\t\nNext\nLeave a Reply Cancel reply\nYou must be logged in to post a comment.\nSearch for:\nFeatured Post\n",
    "Click to read more\nYou must be logged in to view or write comments\.",
    "[0-9,]+ Views",
    "Skip to content",
    "Last Modified Posts",
    "Last Updated:",
    "\(อ่าน [0-9,]+ ครั้ง\)",
    "Recent Comments",
    "«.*?»",
    "< --แสดงทั้งหมด-- >",
    "นโยบายความเป็นส่วนตัว",
    "เงื่อนไขการใช้เว็บไซต์",
    "ตั้งค่าคุกกี้",
    "ท่านยอมรับให้เว็บไซต์นี้จัดเก็บคุกกี้เพื่อประสบการณ์การใช้งานเว็บไซต์ที่ดียิ่งขึ้น",
    "รวมถึงช่วยให้ท่านมีโอกาสได้รับข้อเสนอหรือเนื้อหาที่ตรงตามความสนใจของท่าน",
    "ท่านสามารถดู Privacy Notice ของเว็บไซต์เรา ได้ที่นี่",
    "You may be trying to access this site from a secured browser on the server. Please enable scripts and reload this page.",
    "เผยแพร่: \d\d [ก-๙]+ \d\d\d\d \d\d:\d\d น\.",
    "Last updated: \d\d [ก-๙]+\.[ก-๙]+\. \d\d\d\d \d\d:\d\d น\.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit\.",
    "Search for:",
    "Save my name, email, and website in this browser for the next time I comment",
    "Your email address will not be published. Required fields are marked",
    "Leave a Reply Cancel reply",
    "((?:หน้าหลัก|เข้าสู่ระบบ|หน้าแรก) \|(?: [^\s]+(?:(?: \|)|$|\s))+)",
    "กลับหน้าแรก",
    "ติดต่อเรา",
    "Contact Us",
    "#\w+",
    "ติดต่อผู้ดูแลเว็บไซต์",
    "หากท่านพบว่ามีข้อมูลใดๆที่ละเมิดทรัพย์สินทางปัญญาปรากฏอยู่ในเว็บไซต์โปรดแจ้งให้ทราบ",
    "No related posts",
    "Posted in",
    "((?:Tags:|Tagged|Tag) (?:.{1,40}(?:,|\n|$))+)",
    "ตอบ:",
    "Sort by:",
    "All rights reserved",
    "ความยาวอย่างน้อย",
    "ระบบได้ดำเนินการส่ง OTP",
    "เป็นสมาชิกอยู่แล้ว\?",
    "We use cookies",
    "Cookie Settings",
    "Homeหน้าหลัก",
    "Home หน้าหลัก",
    "ข่าวสารล่าสุด",
    "ปัญหา การใช้งาน",
    "ปัญหาการใช้งาน" "ผู้เขียน",
    "หัวข้อ:",
    "\*\* พร้อมส่ง \*\*",
]

TH_MONTHS = [
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
    "ม\.ค\.",
    "ก\.พ\.",
    "มี\.ค\.",
    "เม\.ย\.",
    "พ\.ค\.",
    "มิ\.ย\.",
    "ก\.ค\.",
    "ส\.ค\.",
    "ก\.ย\.",
    "ต\.ค\.",
    "พ\.ย\.",
    "ธ\.ค\.",
]

CODE_SPECIAL_CHARACTERS = ["\{", "\+", "\}", "/", ":"]

PARTIAL_REMOVAL_KEYWORDS = sorted_by_len(PARTIAL_REMOVAL_KEYWORDS)
