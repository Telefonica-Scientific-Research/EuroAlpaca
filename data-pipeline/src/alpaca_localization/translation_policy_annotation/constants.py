"""Constants used by translation-policy annotation."""

from ..languages import LANG_NAME_BY_TAG as TARGET_LANG_NAME_BY_TAG
from ..languages import TARGET_LANGS


LANG_NAME_BY_TAG = {"en": "English", **TARGET_LANG_NAME_BY_TAG}
DEFAULT_TARGET_LANGS = list(TARGET_LANGS)

POLICY_FIELDS = ("instruction", "input", "output")

SAMPLE_REASON_CODES = {
    "directly_translatable",
    "english_grammar_or_spelling",
    "rhyme_or_sound_constraint",
    "letter_or_character_constraint",
    "word_count_or_line_count_constraint",
    "idiom_or_proverb",
    "wordplay_pun_anagram_palindrome",
    "mnemonic_or_initial_letters",
    "language_specific_task",
    "output_already_violates_instruction",
    "other_not_directly_translatable",
}

CLASSIFICATION_SHORT_LABELS = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "true",
    "false",
    "yes",
    "no",
    "positive",
    "negative",
    "neutral",
    "fiction",
    "non-fiction",
    "interrogative",
    "declarative",
    "imperative",
    "exclamatory",
    "subjective",
    "objective",
    "formal",
    "informal",
    "encyclopedia",
    "newspaper",
    "textbook",
    "online article",
    "political",
    "entertainment",
    "question",
    "statement",
    "command",
    "person",
    "place",
    "animal",
    "carnivore",
    "herbivore",
    "omnivore",
}

CONSTRAINED_LABELS = {
    "0",
    "1",
    "true",
    "false",
    "yes",
    "no",
    "transitive",
    "intransitive",
}
LANGUAGE_ALIAS_TO_TAG = {}
for _tag, _name in LANG_NAME_BY_TAG.items():
    # Avoid ambiguous short tags like "it"/"is" that appear as normal words.
    if len(_tag) >= 3:
        LANGUAGE_ALIAS_TO_TAG[_tag.lower()] = _tag.lower()
    LANGUAGE_ALIAS_TO_TAG[_name.lower()] = _tag.lower()
# Common language-name variants that may appear in instructions.
LANGUAGE_ALIAS_TO_TAG.update(
    {
        "norwegian bokmal": "nob",
        "norwegian bokmål": "nob",
        "norwegian nynorsk": "nno",
        "portuguese (brazil)": "pt",
        "brazilian portuguese": "pt",
        "castilian": "es",
        "espanol": "es",
        "español": "es",
        "catalan": "ca",
        "french": "fr",
        "english": "en",
        "german": "de",
        "italian": "it",
        # Extra aliases used for source/target language detection in prompts.
        # These tags may be outside target_langs and are still useful for
        # deciding whether source/target language constraints are explicit.
        "chinese": "zh",
        "mandarin": "zh",
        "simplified chinese": "zh",
        "traditional chinese": "zh",
        "中文": "zh",
        "japanese": "ja",
        "日本語": "ja",
        "korean": "ko",
        "한국어": "ko",
        "arabic": "ar",
        "hindi": "hi",
        "urdu": "ur",
        "persian": "fa",
        "farsi": "fa",
        "hebrew": "he",
        "turkish": "tr",
        "indonesian": "id",
        "malay": "ms",
        "thai": "th",
        "vietnamese": "vi",
    }
)

__all__ = [
    "LANG_NAME_BY_TAG",
    "POLICY_FIELDS",
    "DEFAULT_TARGET_LANGS",
    "SAMPLE_REASON_CODES",
    "CLASSIFICATION_SHORT_LABELS",
    "CONSTRAINED_LABELS",
    "LANGUAGE_ALIAS_TO_TAG",
]
