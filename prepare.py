with open('gita.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters: {len(text):,}")
print(f"Unique characters: {len(set(text))}")
print(f"\nFirst 500 characters:\n{text[:500]}")
print(f"\nAll unique characters:\n{''.join(sorted(set(text)))}")