from openthaigpt_pretraining_data.web_crawls_mfa.crawl_gov_achievements import (
    process_response,
    get_href,
    href_info,
)

DATE_TITLE_TEST_CASES = [
    {
        "data": '<div class="p-3 col-md-4"><a class="ContentGridItem__Link-sc-jad2bs-0 gaZOqP jsx-2368373130" href="/th/content/กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง-ช่วยบรรเท?page=60b5f529a7079b6e71475d43">title="กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙">กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙</div></a><div class="row"><div class="ContentInfo-sc-1tomowe-0 piEBs col" lang="th"><p class="date">13 ก.ย. 2564</p>',
        "expected_output": [
            {
                "title": "กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙",
                "date": "13 ก.ย. 2564",
            }
        ],
        "data": '<div class="p-3 col-md-4"><a  href="/th/content/tica-รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด?page=60b5f529a7079b6e71475d43"><div clainter" title="TICA รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด">TICA รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด</div><p class="date">2 ก.ย. 2564</p></div><div class="d-inline-block"> <strong class="icon-web-eye"></strong>',
        "expected_output": [
            {
                "title": "TICA รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด",
                "date": "2 ก.ย. 2564",
            }
        ],
    }
]

GET_HREF_TEST_CASES = [
    {
        "data": '<div class="p-3 col-md-4"><a class="ContentGridItem__Link-sc-jad2bs-0 gaZOqP jsx-2368373130" href="/th/content/กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง-ช่วยบรรเท?page=60b5f529a7079b6e71475d43">title="กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙">กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙</div></a><div cl',
        "expected_output": [
            "/th/content/กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง-ช่วยบรรเท?page=60b5f529a7079b6e71475d43"
        ],
        "data": '<dimd-4"><a  href="/th/content/tica-รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด?page=60b5f529a7079b6e71475d43"><div clainter" title="',
        "expected_output": [],
    }
]

HREF_INFO_TEST_CASES = [
    {
        "data": '<div class="ContentDetailstyled__ContentDescription-sc-150bmwg-4 jWrYsI mb-3"><p><span style="font-size: 14pt;">กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙ โดยใช้ช่องทางการทูตในการจัดหาวัคซีนแก่คนไทยและชาวต่างชาติที่อาศัยอยู่ในประเทศไทย พร้อมทั้งสนับสนุนการเข้าถึงวัคซีนอย่างทั่วถึง</span></p></div><div><div class="content-justify-between row align-items-end mt-4 mb-1"><div class="col"><h4 class="m-0">วิดีโอประกอบ</h4></div></div><div class="row"><div class="p-2 col"><div class="slick-slider slick-initialized"><div class="slick-list"><div class="slick-track" style="width:100%;left:0%"><div data-index="0" class="slick-slide slick-active slick-current" tabindex="-1" aria-hidden="false" style="outline:none;width:100%"><div><div class="p-2 clickable" tabindex="-1" style="width:25%;display:inline-block"><div class="SectionVideostyled__CategoryVideoContainer-sc-6xisa7-1 jLiSRT"><div class="LazyImage__LazyImageContainer-sc-10v38ho-0 bItNbh"><span class=" lazy-load-image-background blur" style="background-image:url(https://image.mfa.go.th/mfa/r_50x50/mkKfL2iULZ/0904-64/messageImage_1631518189213.jpg);background-size:100% 100%;color:transparent;display:inline-block"><span class="" style="display:inline-block"></span></span></div><strong class="SectionVideostyled__PlayIcon-sc-6xisa7-2 doJWhd icon-web-play-circled"></strong></div><div title="กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙"',
        "expected_output": [
            "กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙ โดยใช้ช่องทางการทูตในการจัดหาวัคซีนแก่คนไทยและชาวต่างชาติที่อาศัยอยู่ในประเทศไทย พร้อมทั้งสนับสนุนการเข้าถึงวัคซีนอย่างทั่วถึง"
        ],
        "data": '<div class="ContentDetailstyled__ContentDescription-sc-150bmwg-4 jWrYsI mb-3"><div class="kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql ii04i59q">\n<div dir="auto" style="text-align: center;"><strong><span style="font-size: 14pt;">e-Visa ยื่นคำร้องขอวีซ่าออนไลน์ได้ง่าย สะดวก รวดเร็ว ทันใจ บริการใหม่ของกระทรวงการต่างประเทศ</span></strong></div>\n<div dir="auto" style="text-align: center;">&nbsp;</div>\n</div>\n<div class="o9v6fnle cxmmr5t8 oygrvhab hcukyx3x c1et5uql ii04i59q">\n<div dir="auto"><span style="font-size: 14pt;">คุณคริสโตเฟอร์ ไรท์ พาคุณมาพบกับบริการใหม่ของกรมการกงสุล กระทรวงการต่างประเทศ ที่ช่วยให้การยื่นคำร้องขอวีซ่าไม่ยุ่งยากอีกต่อไปด้วยระบบ e-Visa ที่สามารถยื่นคำร้องผ่านระบบออนไลน์ซึ่งเปิดให้บริการแล้ว ๒๑ แห่งใน ๙ ประเทศเพื่ออำนวยความสะดวกให้ชาวต่างชาติที่จะเดินทางเข้ามาในประเทศไทย</sp',
        "expected_output": [
            "e-Visa ยื่นคำร้องขอวีซ่าออนไลน์ได้ง่าย สะดวก รวดเร็ว ทันใจ บริการใหม่ของกระทรวงการต่างประเทศ คุณคริสโตเฟอร์ ไรท์ พาคุณมาพบกับบริการใหม่ของกรมการกงสุล กระทรวงการต่างประเทศ ที่ช่วยให้การยื่นคำร้องขอวีซ่าไม่ยุ่งยากอีกต่อไปด้วยระบบ e-Visa ที่สามารถยื่นคำร้องผ่านระบบออนไลน์ซึ่งเปิดให้บริการแล้ว ๒๑ แห่งใน ๙ ประเทศเพื่ออำนวยความสะดวกให้ชาวต่างชาติที่จะเดินทางเข้ามาในประเทศไทย"
        ],
    }
]


def test_process_response():
    for test_case in DATE_TITLE_TEST_CASES:
        assert process_response(test_case["data"]) == test_case["expected_output"]


def test_get_href():
    for test_case in GET_HREF_TEST_CASES:
        assert get_href(test_case["data"]) == test_case["expected_output"]


def test_href_info():
    for test_case in GET_HREF_TEST_CASES:
        assert href_info(test_case["data"]) == test_case["expected_output"]
