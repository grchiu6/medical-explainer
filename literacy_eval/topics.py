"""Curated list of 30 medical topics for literacy-level evaluation.

These topics are common medical conditions relevant to Indian public
health. Each topic is a question in Hindi asking the model to explain
the condition — what it is, how it happens, or how to manage it.

WHY CURATED (not extracted from a dataset):
For the baseline collection, we want clear, unambiguous medical topics
that a model can explain at any literacy level. HiMed questions are
written for medical professionals (MCQ exam format) and often reference
niche sub-topics. A curated list of common conditions gives us cleaner,
more broadly relevant data to work with.

FORMAT:
Each topic is a dict with:
    - id:       short English identifier (used in filenames and analysis)
    - topic_hi: the question/prompt in Hindi (this is what the model sees)
    - topic_en: English translation (for our reference only, not sent to model)
    - category: broad medical category (for grouping in analysis)
"""

MEDICAL_TOPICS: list[dict] = [
    # --- Metabolic / Endocrine ---
    {
        "id": "diabetes",
        "topic_hi": "मधुमेह (डायबिटीज़) क्या है और यह कैसे होता है?",
        "topic_en": "What is diabetes and how does it occur?",
        "category": "endocrine",
    },
    {
        "id": "thyroid",
        "topic_hi": "थायराइड की बीमारी क्या है और इसके लक्षण क्या हैं?",
        "topic_en": "What is thyroid disease and what are its symptoms?",
        "category": "endocrine",
    },
    {
        "id": "obesity",
        "topic_hi": "मोटापा क्या है और यह स्वास्थ्य को कैसे प्रभावित करता है?",
        "topic_en": "What is obesity and how does it affect health?",
        "category": "metabolic",
    },

    # --- Cardiovascular ---
    {
        "id": "hypertension",
        "topic_hi": "उच्च रक्तचाप (हाई ब्लड प्रेशर) क्या है और इसे कैसे नियंत्रित करें?",
        "topic_en": "What is hypertension and how to control it?",
        "category": "cardiovascular",
    },
    {
        "id": "heart_attack",
        "topic_hi": "दिल का दौरा (हार्ट अटैक) क्यों आता है और इसके लक्षण क्या हैं?",
        "topic_en": "Why does a heart attack happen and what are its symptoms?",
        "category": "cardiovascular",
    },
    {
        "id": "stroke",
        "topic_hi": "ब्रेन स्ट्रोक क्या होता है और इसके चेतावनी संकेत क्या हैं?",
        "topic_en": "What is a brain stroke and what are its warning signs?",
        "category": "cardiovascular",
    },

    # --- Infectious Diseases ---
    {
        "id": "tuberculosis",
        "topic_hi": "तपेदिक (टीबी) क्या है, यह कैसे फैलता है और इसका इलाज कैसे होता है?",
        "topic_en": "What is tuberculosis, how does it spread, and how is it treated?",
        "category": "infectious",
    },
    {
        "id": "dengue",
        "topic_hi": "डेंगू बुखार क्या है और इससे कैसे बचें?",
        "topic_en": "What is dengue fever and how to prevent it?",
        "category": "infectious",
    },
    {
        "id": "malaria",
        "topic_hi": "मलेरिया कैसे होता है और इसका इलाज क्या है?",
        "topic_en": "How does malaria occur and what is its treatment?",
        "category": "infectious",
    },
    {
        "id": "typhoid",
        "topic_hi": "टाइफाइड बुखार क्या है और यह कैसे फैलता है?",
        "topic_en": "What is typhoid fever and how does it spread?",
        "category": "infectious",
    },
    {
        "id": "cholera",
        "topic_hi": "हैजा (कॉलरा) क्या है और इससे कैसे बचाव करें?",
        "topic_en": "What is cholera and how to prevent it?",
        "category": "infectious",
    },
    {
        "id": "covid19",
        "topic_hi": "कोविड-19 क्या है और यह शरीर को कैसे प्रभावित करता है?",
        "topic_en": "What is COVID-19 and how does it affect the body?",
        "category": "infectious",
    },
    {
        "id": "hiv_aids",
        "topic_hi": "एचआईवी/एड्स क्या है, यह कैसे फैलता है और इसका इलाज क्या है?",
        "topic_en": "What is HIV/AIDS, how does it spread, and what is its treatment?",
        "category": "infectious",
    },
    {
        "id": "hepatitis",
        "topic_hi": "हेपेटाइटिस (पीलिया) क्या है और इसके कितने प्रकार हैं?",
        "topic_en": "What is hepatitis and how many types are there?",
        "category": "infectious",
    },

    # --- Respiratory ---
    {
        "id": "asthma",
        "topic_hi": "दमा (अस्थमा) क्या है और इसका प्रबंधन कैसे करें?",
        "topic_en": "What is asthma and how to manage it?",
        "category": "respiratory",
    },
    {
        "id": "pneumonia",
        "topic_hi": "निमोनिया क्या है, इसके लक्षण और इलाज क्या हैं?",
        "topic_en": "What is pneumonia, its symptoms and treatment?",
        "category": "respiratory",
    },

    # --- Gastrointestinal ---
    {
        "id": "diarrhea",
        "topic_hi": "दस्त (डायरिया) क्यों होता है और इसमें क्या करना चाहिए?",
        "topic_en": "Why does diarrhea happen and what should be done?",
        "category": "gastrointestinal",
    },
    {
        "id": "gastric_ulcer",
        "topic_hi": "पेट का अल्सर क्या होता है और इसका इलाज कैसे होता है?",
        "topic_en": "What is a gastric ulcer and how is it treated?",
        "category": "gastrointestinal",
    },

    # --- Musculoskeletal ---
    {
        "id": "arthritis",
        "topic_hi": "गठिया (आर्थराइटिस) क्या है और इसके लक्षण क्या हैं?",
        "topic_en": "What is arthritis and what are its symptoms?",
        "category": "musculoskeletal",
    },

    # --- Renal ---
    {
        "id": "kidney_stones",
        "topic_hi": "गुर्दे की पथरी (किडनी स्टोन) क्यों बनती है और इसका इलाज क्या है?",
        "topic_en": "Why do kidney stones form and what is the treatment?",
        "category": "renal",
    },
    {
        "id": "kidney_disease",
        "topic_hi": "किडनी (गुर्दे) की बीमारी क्या है और इसके शुरुआती लक्षण क्या हैं?",
        "topic_en": "What is kidney disease and what are its early symptoms?",
        "category": "renal",
    },

    # --- Hematological ---
    {
        "id": "anemia",
        "topic_hi": "खून की कमी (एनीमिया) क्या है और यह क्यों होती है?",
        "topic_en": "What is anemia and why does it happen?",
        "category": "hematological",
    },

    # --- Neurological ---
    {
        "id": "epilepsy",
        "topic_hi": "मिर्गी (एपिलेप्सी) क्या है और इसका इलाज कैसे होता है?",
        "topic_en": "What is epilepsy and how is it treated?",
        "category": "neurological",
    },
    {
        "id": "migraine",
        "topic_hi": "माइग्रेन क्या है और इसे सामान्य सिरदर्द से कैसे अलग पहचानें?",
        "topic_en": "What is a migraine and how to differentiate it from a normal headache?",
        "category": "neurological",
    },

    # --- Mental Health ---
    {
        "id": "depression",
        "topic_hi": "अवसाद (डिप्रेशन) क्या है और इसके लक्षण क्या हैं?",
        "topic_en": "What is depression and what are its symptoms?",
        "category": "mental_health",
    },

    # --- Dermatological ---
    {
        "id": "eczema",
        "topic_hi": "एक्ज़िमा (त्वचा की सूजन) क्या है और इसका इलाज कैसे होता है?",
        "topic_en": "What is eczema and how is it treated?",
        "category": "dermatological",
    },

    # --- Maternal / Reproductive ---
    {
        "id": "pregnancy_care",
        "topic_hi": "गर्भावस्था (प्रेगनेंसी) में किन बातों का ध्यान रखना चाहिए?",
        "topic_en": "What precautions should be taken during pregnancy?",
        "category": "maternal",
    },

    # --- Nutritional ---
    {
        "id": "malnutrition",
        "topic_hi": "कुपोषण क्या है और बच्चों में यह क्यों होता है?",
        "topic_en": "What is malnutrition and why does it occur in children?",
        "category": "nutritional",
    },

    # --- Ophthalmological ---
    {
        "id": "cataract",
        "topic_hi": "मोतियाबिंद क्या है और इसका ऑपरेशन कब करवाना चाहिए?",
        "topic_en": "What is a cataract and when should surgery be done?",
        "category": "ophthalmological",
    },

    # --- Oncological ---
    {
        "id": "cancer_awareness",
        "topic_hi": "कैंसर क्या है, इसके सामान्य लक्षण क्या हैं और जल्दी पहचान क्यों ज़रूरी है?",
        "topic_en": "What is cancer, what are common symptoms, and why is early detection important?",
        "category": "oncological",
    },
]


def load_topics(sample_size: int | None = None) -> list[dict]:
    """Return the curated medical topics list.

    Args:
        sample_size: If provided, return only the first N topics.
                     (We don't randomize here — topics are already in a
                     logical order by category, and deterministic ordering
                     makes resume support simpler.)

    Returns:
        List of topic dicts with keys: id, topic_hi, topic_en, category.
    """
    topics = MEDICAL_TOPICS
    if sample_size is not None:
        topics = topics[:sample_size]
    return topics
