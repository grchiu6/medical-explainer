"""Tests for literacy_eval/register/clean_sources.py — removing non-prose from OCR'd pages.

Strings are taken from real OCR output of NCERT ehev1/ehap1 and ICMR Patrika pages.
"""

from pathlib import Path

from literacy_eval.register import clean_sources as C

PROSE = "ज्योति यह जानने को उत्सुक थी कि क्या वायु में भी जल की उपस्थिति होती है? और क्या बर्फ जल का ही एक रूप है?"
ICMR_PROSE = ("भारत की आधी आबादी प्रजननशील आयु वर्ग के अन्तर्गत है और स्वास्थ्य सुविधाएं देश के सुदूर क्षेत्रों में "
              "पहुंचे, परन्तु ऐसा नहीं हो पाने के कारण इन वर्गों में मृत्यु दर बहुत अधिक है और रोग का बोझ भी अधिक है।")


def para(text, conf=90.0):
    return C.Paragraph(text=text, conf=conf)


def test_strip_furniture_removes_indd_and_date_stamps():
    assert C.strip_furniture("Chapter l.indd 5 हम पहले से जानते हैं 27-00-2026 4:7:") == "हम पहले से जानते हैं"


def test_teacher_page_dropped_whole():
    drops = C.DropCounter()
    assert C.clean_page("ncert", [para("शिक्षकों के लिए"), para(PROSE)], drops) == []
    assert drops["page:ncert_teacher"] == 1


def test_teacher_box_paragraph_dropped_but_story_kept():
    drops = C.DropCounter()
    box = "विद्यार्थियों को उनके घर या विद्यालय के आस-पास के जल-निकायों के विषय में सोचने के लिए प्रोत्साहित कीजिए"
    assert C.clean_page("ncert", [para(box), para(PROSE)], drops) == [PROSE]
    assert drops["para:ncert_teacher"] == 1


def test_exercise_sentences_dropped_narrative_question_kept():
    drops = C.DropCounter()
    text = ("नीचे दिए गए चित्रों में से मीठे जल के विभिन्न स्रोतों को पहचानिए और उनके नाम लिखिए। "
            "क्या तुम्हारे घर पर किसी ने तुम्हें नाक बंद करके दवाई पीने को कहा है? "
            "आइए, हम एक सरल गतिविधि के द्वारा इसे समझते हैं। " + PROSE)
    assert C.drop_exercise_sentences(text, drops) == PROSE
    assert drops["sentence:ncert_exercise"] == 3


def test_low_confidence_short_and_non_devanagari_paragraphs_dropped():
    drops = C.DropCounter()
    paragraphs = [para(PROSE, conf=40), para("जल के विभिन्न रूप"), para("Smith J, et al. Lancet 2015; 12: 33-40 रोग")]
    assert C.clean_page("ncert", paragraphs, drops) == []
    assert drops["para:low_confidence"] == 1 and drops["para:short"] == 1 and drops["para:non_devanagari"] == 1


def test_icmr_editorial_and_non_medical_pages_dropped():
    drops = C.DropCounter()
    assert C.clean_page("icmr", [para("संपादक मंडल"), para(ICMR_PROSE)], drops) == []
    assert C.clean_page("icmr", [para("संस्थान में वृक्षारोपण कार्यक्रम बड़े उत्साह से मनाया गया और सभी ने भाग लिया।")], drops) == []
    assert drops["page:icmr_non_article"] == 1 and drops["page:icmr_non_medical"] == 1


def test_icmr_article_kept_and_author_line_dropped():
    drops = C.DropCounter()
    author = "डॉ. रमेश कुमार, वैज्ञानिक 'ई', राष्ट्रीय पोषण संस्थान, हैदराबाद"
    assert C.clean_page("icmr", [para(author), para(ICMR_PROSE)], drops) == [ICMR_PROSE]
    assert drops["para:icmr_author_line"] == 1


def test_keep_file_excludes_front_matter_answers_and_non_biology_class10():
    assert not C.keep_file("ncert", "ncert/ehev1/ehev1ps.pdf")
    assert not C.keep_file("ncert", "ncert/jhsc1/jhsc1an.pdf")
    assert not C.keep_file("ncert", "ncert/jhsc1/jhsc101.pdf")      # chemistry
    assert C.keep_file("ncert", "ncert/jhsc1/jhsc105.pdf")          # जैव प्रक्रम
    assert C.keep_file("ncert", "ncert/ehap1/ehap103.pdf")
    assert C.keep_file("icmr", "icmr/april.pdf")


def test_page_paragraphs_reads_tesseract_tsv(tmp_path: Path):
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    rows = [
        "1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t\n",
        "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tपहला\n",
        "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t80\tपैरा\n",
        "5\t1\t2\t1\t1\t1\t0\t0\t1\t1\t60\tदूसरा\n",
    ]
    tsv = tmp_path / "p001.tsv"
    tsv.write_text(header + "".join(rows), encoding="utf-8")
    paragraphs = C.page_paragraphs(tsv)
    assert [p.text for p in paragraphs] == ["पहला पैरा", "दूसरा"]
    assert paragraphs[0].conf == 85.0


def test_imperatives_to_student_are_exercises_but_for_and_should_are_not():
    assert C.is_exercise_sentence("अपने विद्यालय के पास किसी वृक्ष को देखिए और उसका चयन कीजिए।")
    assert C.is_exercise_sentence("चित्रों को देखकर अंदाजा लगाओ कि बीज कैसे बिखर रहे हैं?")
    assert C.is_exercise_sentence("आपको इस परेड में सबसे अच्छा क्या लगा?")
    assert not C.is_exercise_sentence("पौधों को बढ़ने के लिए पानी चाहिए।")
    assert not C.is_exercise_sentence("इसलिए हमें जल बचाना चाहिए।")
    assert not C.is_exercise_sentence("उन्होंने कई प्रयोग किए और परिणाम दिए।")


def test_icmr_event_report_page_dropped():
    drops = C.DropCounter()
    report = ("रोग नियंत्रण पर बैठक हुई। स्वास्थ्य मंत्री ने उद्घाटन किया। मुख्य अतिथि ने उपचार और "
              "रोगी देखभाल पर बात की और सम्मेलन में चिकित्सा विशेषज्ञ आए।")
    assert C.clean_page("icmr", [para(report)], drops) == []
    assert drops["page:icmr_event_report"] == 1


def test_icmr_sentences_naming_publisher_or_people_dropped():
    drops = C.DropCounter()
    text = "आई सी एम आर ने यह अध्ययन किया। डॉ. शर्मा ने बताया। " + ICMR_PROSE
    assert C.drop_icmr_source_sentences(text, drops) == ICMR_PROSE
    assert drops["sentence:icmr_source_marker"] == 2


def test_sentence_final_imperative_before_danda_is_still_an_exercise():
    # Regression: the danda is inside the Devanagari Unicode block, so "...कीजिए।" was not matched.
    for sentence in ["जडों को धो लीजिए।", "स्लाइड को रखिए।", "कवरस्लिप को धीरे से दबाइए।",
                     "सूक्ष्मदर्शी द्वारा स्लाइड का अवलोकन कीजिए।", "चित्रों को देखो।", "तुम्हें क्या लगा?"]:
        assert C.is_exercise_sentence(sentence), sentence
    assert not C.is_exercise_sentence("पौधों को बढ़ने के लिए पानी चाहिए।")
    assert not C.is_exercise_sentence("इसलिए हमें जल बचाना चाहिए।")
