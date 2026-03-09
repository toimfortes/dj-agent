# Title Cleanup Rules

Applied in order:

```python
import re, html

def cleanup_title(title, artist=""):
    original = title
    changes = []

    # 1. Decode HTML entities: &#39; -> ', &amp; -> &
    decoded = html.unescape(title)
    if decoded != title:
        changes.append("decoded HTML entities")
        title = decoded

    # 2. Strip trailing/leading whitespace
    title = title.strip()

    # 3. Collapse double spaces
    title = re.sub(r'  +', ' ', title)

    # 4. Remove website watermarks
    watermark_patterns = [
        r'\s*-?\s*www\.\S+',
        r'\s*\[?\w+\.com\]?\s*$',
        r'\s*\((?:promodj|djsoundtop|electronicfresh|zipdj)\.\w+\)',
        r'\s+djsoundtop\.\w+',
        r'\s+electronicfresh\.\w+',
        r'\s*#VKUSMUZ\s*',
    ]
    for pattern in watermark_patterns:
        cleaned = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
        if cleaned != title:
            changes.append("removed website watermark")
            title = cleaned

    # 5. Remove file extensions from titles
    title = re.sub(r'\.(mp3|flac|wav|aiff|m4a|aac|ogg)$', '', title, flags=re.IGNORECASE).strip()

    # 6. Remove trailing " - KEY - BPM" (e.g. " - 3A - 124")
    title = re.sub(r'\s*-\s*\d{1,2}[AB]\s*-\s*\d{2,3}\s*$', '', title).strip()

    # 7. Remove trailing Beatport/store IDs (e.g. "-62381251")
    title = re.sub(r'-\d{8,}\s*$', '', title).strip()

    # 8. Convert filename-style underscores to spaces (if 3+ underscores)
    if title.count('_') >= 3 and '-' in title:
        parts = title.split('-')
        cleaned_parts = [p.strip().replace('_', ' ').strip() for p in parts]
        cleaned_parts = [p for p in cleaned_parts if not re.match(r'^\d{6,}$', p)]
        if artist and cleaned_parts and cleaned_parts[0].lower().replace(' ', '') == artist.lower().replace(' ', ''):
            cleaned_parts = cleaned_parts[1:]
        title = ' - '.join(cleaned_parts).strip(' -')

    # 9. Remove [PRO FRONT] or similar download source tags
    title = re.sub(r'\[PRO FRONT\]\s*', '', title, flags=re.IGNORECASE).strip()

    # 10. Remove trailing bare year stamps (e.g. " 1996", " 2025")
    title = re.sub(r'\s+\d{4}\s*$', '', title).strip()

    # 11. Remove trailing junk: (1), MASTER, Free DL, codes like sm1303/MH1644, JH
    junk_patterns = [
        r'\s*\(\d+\)\s*$',
        r'\s+MASTER\s*$',
        r'\s+Free\s+DL\s*$',
        r'\s+[A-Z]{0,3}\d{3,5}\s*$',
        r'\s+JH\s*$',
    ]
    for pattern in junk_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()

    # 12. Final trim
    title = title.strip(' -\u2013')

    return title, changes
```

## Smart Title Case

```python
def smart_title_case(title):
    lowercase_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for',
                       'in', 'on', 'at', 'to', 'of', 'by', 'vs', 'x',
                       'feat', 'feat.', 'ft', 'ft.'}
    uppercase_words = {'dj', 'sos', 'uk', 'usa', 'id', 'vip', 'ep', 'lp',
                       'og', 'hd', 'bpm', 'ok', 'tv', 'ii', 'iii', 'iv',
                       'nyc', 'la', 'sf', 'dc'}

    def case_word(word, is_first):
        lower = word.lower()
        if any(c in word for c in '/'):
            return word
        if lower in uppercase_words:
            return word.upper()
        if lower in lowercase_words and not is_first:
            return lower
        if word:
            return word[0].upper() + word[1:]
        return word

    words = title.split(' ')
    cased = [case_word(w, i == 0) for i, w in enumerate(words)]
    return ' '.join(cased)
```
