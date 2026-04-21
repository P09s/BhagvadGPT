with open('gita.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Raw characters: {len(text):,}")

start_marker = "SONG CELESTIAL"
end_marker = "End of the Project Gutenberg"

str_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if str_idx == -1:
    start_marker = "BOOK THE FIRST"
    str_idx = text.find(start_marker)

text = text[str_idx:end_idx].strip()
print(f"After stripping data in between header and footer: {len(text):,}")

# ----- Cleaning noisy characters----------

replacement = {
     '\u2014': '-',    # em dash  →  hyphen
    '\u2018': "'",    # left single quote
    '\u2019': "'",    # right single quote
    '\u201c': '"',    # left double quote
    '\u201d': '"',    # right double quote
    '\u2022': '',     # bullet
    '\u2122': '',     # trademark
    '\u00c2': 'A',   # Â
    '\u00ce': 'I',   # Î
    '\u00e2': 'a',   # â
    '\u00ee': 'i',   # î
    '\r':     '',     # carriage return
}

for old, new in replacement.items():
    text = text.replace(old, new)

# removing whitespace lines
lines = [line for line in text.split('\n') if line.strip()]
text = '\n'.join(lines)

print(f"After cleaning:              {len(text):,}")
print(f"Unique characters now:       {len(set(text))}")
print(f"\nAll chars: {''.join(sorted(set(text)))}")
print(f"\n--- First 800 characters of CLEAN text ---\n{text[:800]}")

# save clean data

with open('gita_clean.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("\n saved to git_clean.txt")