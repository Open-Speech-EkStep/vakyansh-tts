import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def basic_indic_cleaners(text):
    """Basic pipeline that collapses whitespace without transliteration."""
    text = collapse_whitespace(text)
    return text
