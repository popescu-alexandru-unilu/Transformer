with open('../data/text8', 'rb') as f:
    raw = f.read()

# Try detecting the encoding
import chardet
detected_encoding = chardet.detect(raw)
print(f"Detected encoding: {detected_encoding}")

# If it's not UTF-8, convert it
if detected_encoding['encoding'] != 'utf-8':
    with open('../data/text8', 'r', encoding=detected_encoding['encoding']) as f:
        content = f.read()
    with open('../data/text8', 'w', encoding='utf-8') as f:
        f.write(content)
    print("File converted to UTF-8 encoding.")
