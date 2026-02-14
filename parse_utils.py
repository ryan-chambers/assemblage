def strip_and_fix_chars(s: str) -> str:
    s = s.strip()
    s = s.replace('–', '-')  # en-dash to hyphen
    s = s.replace('\u2019', "'")  # right single quotation mark to apostrophe
    return s
