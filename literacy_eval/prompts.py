"""Prompt templates for literacy-level adapted Hindi medical explanations.

This module defines three literacy levels and builds prompts that instruct
a model to explain a medical topic at a specific linguistic register.

KEY DESIGN DECISIONS:

1. INSTRUCTIONS IN HINDI: Since the research question is about serving
   Hindi-speaking populations, the prompt instructions are in Hindi.
   Real-world users would prompt in Hindi, so the baseline should
   reflect that scenario. If a model can only adapt register with
   English instructions but not Hindi, that's a finding worth capturing.

2. THREE LITERACY LEVELS: Low (5th grade), Mid (10th grade), High
   (medical professional). These map roughly to primary education,
   secondary education, and professional training — the three bands
   most health-literacy frameworks use.

3. FREE-FORM RESPONSE: No output structure imposed. We want to see
   what the model naturally produces when asked to explain at a given
   level. Structure constraints can be added later if needed for analysis.

4. CONCRETE LINGUISTIC GUIDANCE: Each level specifies vocabulary,
   sentence structure, and jargon expectations in concrete terms, not
   just "simple" or "complex." This gives the model actionable
   constraints and makes it easier to later assess whether it followed them.
"""

from __future__ import annotations


# Each level has:
#   - id:          short identifier (used in filenames, CLI args, CSV columns)
#   - label_hi:    Hindi label for the level
#   - label_en:    English label for reference
#   - instruction: Hindi prompt instruction describing the target register
#
# The instruction text is written to be self-contained — it describes
# the target audience, vocabulary constraints, sentence complexity, and
# any special requirements (e.g., defining jargon when used).

LITERACY_LEVELS: dict[str, dict] = {
    "low": {
        "id": "low",
        "label_hi": "बुनियादी (कक्षा 5)",
        "label_en": "Basic (5th grade)",
        "instruction": (
            "आप एक चिकित्सा शिक्षक हैं। नीचे दिए गए चिकित्सा विषय को हिंदी में समझाएं।\n"
            "\n"
            "लक्षित श्रोता: एक ऐसा व्यक्ति जिसने केवल पांचवीं कक्षा तक पढ़ाई की है।\n"
            "\n"
            "भाषा संबंधी निर्देश:\n"
            "- केवल सरल, रोज़मर्रा की हिंदी का प्रयोग करें (बोलचाल की भाषा)\n"
            "- किसी भी चिकित्सा या अंग्रेज़ी शब्द का प्रयोग न करें\n"
            "- छोटे और सीधे वाक्य लिखें (5-10 शब्दों के)\n"
            "- रोज़मर्रा की ज़िंदगी से उदाहरण देकर समझाएं\n"
            "- ऐसे समझाएं जैसे आप किसी गाँव के व्यक्ति को बता रहे हों\n"
        ),
    },
    "mid": {
        "id": "mid",
        "label_hi": "मध्यम (कक्षा 10)",
        "label_en": "Intermediate (10th grade)",
        "instruction": (
            "आप एक चिकित्सा शिक्षक हैं। नीचे दिए गए चिकित्सा विषय को हिंदी में समझाएं।\n"
            "\n"
            "लक्षित श्रोता: एक ऐसा व्यक्ति जिसने दसवीं कक्षा तक पढ़ाई की है।\n"
            "\n"
            "भाषा संबंधी निर्देश:\n"
            "- सामान्य हिंदी का प्रयोग करें, कुछ चिकित्सा शब्दों के साथ\n"
            "- जब कोई चिकित्सा शब्द प्रयोग करें तो उसका सरल अर्थ भी बताएं\n"
            "- मध्यम लंबाई के वाक्य लिखें\n"
            "- बीमारी के कारण, लक्षण और उपचार के बारे में व्यवस्थित रूप से बताएं\n"
            "- शरीर की बुनियादी संरचना का संदर्भ दे सकते हैं\n"
        ),
    },
    "high": {
        "id": "high",
        "label_hi": "विशेषज्ञ (चिकित्सा पेशेवर)",
        "label_en": "Expert (Medical professional)",
        "instruction": (
            "आप एक चिकित्सा विशेषज्ञ हैं। नीचे दिए गए चिकित्सा विषय को हिंदी में समझाएं।\n"
            "\n"
            "लक्षित श्रोता: एक चिकित्सा छात्र या स्वास्थ्य कार्यकर्ता।\n"
            "\n"
            "भाषा संबंधी निर्देश:\n"
            "- तकनीकी चिकित्सा शब्दावली का प्रयोग करें\n"
            "- रोग की पैथोफिज़ियोलॉजी (रोग-क्रियाविज्ञान) का वर्णन करें\n"
            "- जटिल वाक्य संरचना का प्रयोग कर सकते हैं\n"
            "- नैदानिक मानदंड, जोखिम कारक और उपचार प्रोटोकॉल शामिल करें\n"
            "- मान लें कि पाठक को शरीर विज्ञान (anatomy) और क्रिया विज्ञान (physiology) की "
            "बुनियादी जानकारी है\n"
        ),
    },
}

# Valid level IDs — used for CLI argument validation
VALID_LEVELS = sorted(LITERACY_LEVELS.keys())


def build_explanation_prompt(topic_hi: str, literacy_level: str) -> str:
    """Build a full prompt for a medical topic at a given literacy level.

    Combines the level-specific instruction with the medical topic to
    create a complete prompt ready to send to any model's get_answer().

    Args:
        topic_hi: The medical topic question in Hindi.
        literacy_level: One of 'low', 'mid', 'high'.

    Returns:
        A formatted prompt string ready for a model.

    Raises:
        ValueError: If literacy_level is not one of the valid levels.

    Example:
        >>> prompt = build_explanation_prompt(
        ...     "मधुमेह (डायबिटीज़) क्या है?", "low"
        ... )
        >>> print(prompt[:50])
        आप एक चिकित्सा शिक्षक हैं। नीचे दिए गए चिकित
    """
    if literacy_level not in LITERACY_LEVELS:
        raise ValueError(
            f"Unknown literacy level '{literacy_level}'. "
            f"Valid levels: {VALID_LEVELS}"
        )

    level = LITERACY_LEVELS[literacy_level]

    return (
        f"{level['instruction']}\n"
        f"चिकित्सा विषय: {topic_hi}\n"
        f"\n"
        f"हिंदी में व्याख्या:"
    )
