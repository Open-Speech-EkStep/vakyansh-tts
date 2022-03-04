import re

from unidecode import unidecode
from .numbers import normalize_numbers




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


def english_cleaner(text):
    text = text.lower().replace('‘','\'').replace('’','\'')
    return text


def lowercase(text):
    return text.lower()

def convert_to_ascii(text):
    return unidecode(text)

def expand_numbers(text):
    return normalize_numbers(text)

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'missus'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
  ('pvt', 'private'),
  ('rs', 'Rupees')
]]






def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
