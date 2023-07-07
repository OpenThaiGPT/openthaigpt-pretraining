from src.data.openthaigpt_pretraining_data.pantip.clean_data_pantip3g import clean_data

CLEAN_HTML_TAGS_TEST_CASES = [
    {"data": "<h1>Hello world</h1>", "expected_output": "Hello world"},
    {"data": "<h1>Hello world", "expected_output": "Hello world"},
    {"data": "Hello world", "expected_output": "Hello world"},
    {
        "data": "addictiondh\nhttps://www.youtube.com/watch?v=HUHUHUHUASDKasd\n\n",
        "expected_output": "addictiondh\n website",
    },
    {
        "data": '<a target="_blank" href="http://pantipopop.comunity/topic/31XXXXX1" rel="nofollow" >รวมวิธี How to แนะนำการใช้งาน Panptipopop โฉมใหม่</a>',
        "expected_output": "รวมวิธี How to แนะนำการใช้งาน Panptipopop โฉมใหม่",
    },
    {
        "data": '<a>เช่น ถ้าจะถ้าเกี่ยวกับ truemaikayab ก็ติด tag [truemaikayab] [Spoil] คลิกเพื่อดูข้อความที่ซ่อนไว้</a><div class="spoil-style" style="display:none;">ไอคอนกดให้คะแนน</div>',
        "expected_output": "เช่น ถ้าจะถ้าเกี่ยวกับ truemaikayab ก็ติด tag [truemaikayab] ไอคอนกดให้คะแนน",
    },
    {
        "data": "ไม่กินhttp://forumnaja.monkeygadget.com/detail.php?id=2975112456890\n\nผมกังวลว่าลิงจะมากินผม",
        "expected_output": "ไม่กิน website\n\nผมกังวลว่าลิงจะมากินผม",
    },
    {
        "data": "\nhttps://www.youtube.com/watch?v=2qrnjkfsali02qASSZdfhkjgkd&feature=youtube_gdata_player  ประเภท",
        "expected_output": "\nwebsite  ประเภท",
    },
    {
        "data": "An Error Was Encountered\nFind of data into MongoDB failed: localhost:27017: DBClientBase::findN: transport error: mongod3.local:27017 ns: admin.$cmd query: { setShardVersion: &quot;&quot;, init: true, configdb: &quot;img1.local:27019,img2.local:27019,central.local:27019&quot;, serverID: ObjectId('50d6b05152fc473fe11946c7'), authoritative: true, $auth:  }</div>\n\nอันนี้เวลาเปิดดูบางห้อง\n\n",
        "expected_output": "an อันนี้เวลาเปิดดูบางห้อง",
    },
    {
        "data": '<img class="img-in-emotion" title="ยิ้ม" alt="ยิ้ม" src="http://ptctcto.infomation/emoticons/emoticon-smilinggggg_na_ja.png"/>ยิ้มอร่อยดี',
        "expected_output": "ยิ้มอร่อยดี",
    },
    {
        "data": "tag [วิทยาศาสตร์] ถูกใช้บ่อยมากในการขอความเห็นเชิงวิทยาศาสตร์",
        "expected_output": "tag [วิทยาศาสตร์] ถูกใช้บ่อยมากในการขอความเห็นเชิงวิทยาศาสตร์",
    },
    {
        "data": "[img]http://www.quietplease.com/images/products/picture.jpg[/img]finish",
        "expected_output": "finish",
    },
    {"data": "www.yohoho.pirate Hohoho~", "expected_output": "website Hohoho~"},
    {
        "data": "\t\t\t\t\t\t\t\t\t\t\t\t\t\t So many tab \\t",
        "expected_output": "So many tab \\t",
    },
]


def test_clean_data():
    for test_case in CLEAN_HTML_TAGS_TEST_CASES:
        assert clean_data(test_case["data"]) == test_case["expected_output"]
