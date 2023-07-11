from openthaigpt_pretraining_data.web_crawls_mfa.crawl_gov_achievements import (
    process_response,
)

CRAWL_GOV_ACHIEVEMENT_TEST_CASES = [
    {
        "data": '<div class="p-3 col-md-4"><a class="ContentGridItem__Link-sc-jad2bs-0 gaZOqP jsx-2368373130" href="/th/content/กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง-ช่วยบรรเท?page=60b5f529a7079b6e71475d43">',
        "expected_output": "กระทรวงการต่างประเทศร่วมเป็นอีกหนึ่งพลัง ช่วยบรรเทาผลกระทบจากโควิด-๑๙ โดยใช้ช่องทางการทูตในการจัดหาวัคซีนแก่คนไทยและชาวต่างชาติที่อาศัยอยู่ในประเทศไทย พร้อมทั้งสนับสนุนการเข้าถึงวัคซีนอย่างทั่วถึง",
        "data": '<div class="p-3 col-md-4"><a  href="/th/content/tica-รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด?page=60b5f529a7079b6e71475d43"><div cla',
        "expected_output": 'TICA รวมพลังน้ำใจไทย-ประเทศเพื่อนบ้านต้านภัยโควิด เพราะความร่วมมือ...เป็นกุญแจสำคัญของการพัฒนาที่ยั่งยืน กระทรวงการต่างประเทศโดยกรมความร่วมมือระหว่างประเทศ จัดทำโครงการ "ความร่วมมือกับประเทศเพื่อนบ้าน" เพื่อรับมือวิกฤตโควิด-๑๙ ในการมอบองค์ความรู้ อุปกรณ์ทางการแพทย์ และสนับสนุนการเตรียมความพร้อมในทุกสถานการณ์ เพื่อสร้างสาธารณสุขที่ดีให้แก่มิตรประเทศเพื่อนบ้านและส่งเสริมความสัมพันธ์ที่ดีระหว่างกัน และก้าวผ่านวิกฤตในครั้งนี้ไปด้วยกัน',
    }
]


def process_response():
    for test_case in CRAWL_GOV_ACHIEVEMENT_TEST_CASES:
        assert process_response(test_case["data"]) == test_case["expected_output"]
