from openthaigpt_pretraining_data.web_crawls_mcot.crawl_mcot import (
    process_title_date,
    get_href,
    href_info,
)

from openthaigpt_pretraining_data.web_crawls_mcot.getpages_mcot import (
    get_main_href,
    page_number_response,
)

TITLE_DATE_TESTCASES = [
    {
        "data": '<header class="entry-header"><h2 class="entry-title"><a href="https://tna.mcot.net/politics-1209842" title="Permalink to สั่งหน่วยทหาร เร่งช่วยระนอง และจังหวัดที่น้ำท่วม" rel="bookmark">สั่งหน่วยทหาร เร่งช่วยระนอง และจังหวัดที่น้ำท่วม</a></h2><div class="time">19/07/2566<span>16:39</span></div>',
        "expected_output": [
            {
                "title": "สั่งหน่วยทหาร เร่งช่วยระนอง และจังหวัดที่น้ำท่วม",
                "date": "19/07/2566",
            }
        ],
        "data": '<header class="entry-header"><h2 class="entry-title"><a href="https://tna.mcot.net/politics-1209676" title="Permalink to รทสช.คุยได้ทุกพรรค ยกเว้นพรรคแก้112" rel="bookmark">รทสช.คุยได้ทุกพรรค ยกเว้นพรรคแก้112</h2><div class="time">19/07/2566',
        "expected_output": [
            {"title": "รทสช.คุยได้ทุกพรรค ยกเว้นพรรคแก้112", "date": "19/07/2566"}
        ],
    }
]

NEWS_HREF_TESTCASES = [
    {
        "data": '<h2 class="entry-title"><a href="https://tna.mcot.net/politics-1209675" title="Permalink to',
        "expected_output": ["https://tna.mcot.net/politics-1209675"],
        "data": '<h2 class="entry-title"><a href="https://tna.mcot.net/politics-1208880"<a href="https://tna.mcot.net/region-1196789" title="Permalink to พื้นที่อิสระของคนรักศิลปะ ที่บ้านศิลปิน"',
        "expected_output": ["https://tna.mcot.net/politics-1208880"],
    }
]

HREF_INFO_TESTCASES = [
    {
        "data": '<div class="entry-content"><p><strong>รัฐสภา 19 ก.ค.-“ธนกร” ยืนยันไม่ร่วมรัฐบาลกับพรรคแก้ ม.112 ส่วนพรรคอื่นคุยกันได้ &nbsp;&nbsp;เชื่อก้าวไกลไม่ลดเพดาน เพราะเดินมาไกลแล้ว</strong></p>',
        "expected_output": [
            "รัฐสภา 19 ก.ค.-“ธนกร” ยืนยันไม่ร่วมรัฐบาลกับพรรคแก้ ม.112 ส่วนพรรคอื่นคุยกันได้ \xa0\xa0เชื่อก้าวไกลไม่ลดเพดาน เพราะเดินมาไกลแล้ว"
        ],
        "data": '<div class="entry-content"><p>ผู้สื่อข่าวรายงานว่า&nbsp; ระหว่างที่สมาชิกรัฐสภายังคงอภิปรายตาม ข้อบังคับที่ 151 &nbsp;เพื่อตีความข้อบังคับที่ 41 &nbsp;&nbsp;ว่าการเสนอชื่อนายพิธา ลิ้มเจริญรัตน์&nbsp; &nbsp;ในการเลือกนายกรัฐมนตรีวันนี้&nbsp;&nbsp; ถือเป็นการเสนอญัตติซ้ำหรือไม่<div class="related-posts"><h2 class="s-title"><span>ข่าวที่เกี่ยวข้อง</span></h2><ul><li><h2 class="entry-title">บีบีซีชี้ “พิธา”',
        "expected_output": [
            "ผู้สื่อข่าวรายงานว่า\xa0 ระหว่างที่สมาชิกรัฐสภายังคงอภิปรายตาม ข้อบังคับที่ 151 \xa0เพื่อตีความข้อบังคับที่ 41 \xa0\xa0ว่าการเสนอชื่อนายพิธา ลิ้มเจริญรัตน์\xa0 \xa0ในการเลือกนายกรัฐมนตรีวันนี้\xa0\xa0 ถือเป็นการเสนอญัตติซ้ำหรือไม่"
        ],
    }
]


MAIN_HREF_TESTCASES = [
    {
        "data": '<div class="menu-main-menu-container"><li id="menu-item-455201" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-has-children menu-item-455201"><a href="https://tna.mcot.net/category/politics">การเมือง</a>',
        "expected_output": ["https://tna.mcot.net/category/politics"],
        "data": '<div class="menu-main-menu-container">4"><a href="https://tna.mcot.net/category/business/settrade">ตลาดหุ้นไทย</a></li><li id="menu-item-488421"><a href="https://tna.mcot.net/category/business/special-business">เศรษฐกิจ สกู๊ปพิเศษ</a></li>',
        "expected_output": [
            "https://tna.mcot.net/category/business/settrade",
            "https://tna.mcot.net/category/business/special-business",
        ],
    }
]

PAGE_NO_TESTCASES = [
    {
        "data": '<div class="content-pagination"><span aria-current="page" class="page-numbers current">1</span><a class="page-numbers" href="https://tna.mcot.net/category/food-travel/page/2">2</a><a class="page-numbers" href="https://tna.mcot.net/category/food-travel/page/12">12</a><a class="next page-numbers" href="https://tna.mcot.net/category/food-travel/page/2"><i class="si-angle-right"></i></a></div>',
        "expected_output": 12,
        "data": '<div class="content-pagination"></div>',
        "expected_output": 1,
        "data": '<header class="page-header"><h1 class="page-title entry-title hide">World Pulse</h1></header>',
        "expected_output": 0,
    }
]


def test_title_date():
    for test_case in TITLE_DATE_TESTCASES:
        assert process_title_date(test_case["data"]) == test_case["expected_output"]


def test_news_href():
    for test_case in NEWS_HREF_TESTCASES:
        assert get_href(test_case["data"]) == test_case["expected_output"]


def test_href_info():
    for test_case in HREF_INFO_TESTCASES:
        assert href_info(test_case["data"]) == test_case["expected_output"]


def test_main_href():
    for test_case in MAIN_HREF_TESTCASES:
        assert get_main_href(test_case["data"]) == test_case["expected_output"]


def test_get_page():
    for test_case in PAGE_NO_TESTCASES:
        assert page_number_response(test_case["data"]) == test_case["expected_output"]
