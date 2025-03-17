import re

# Clean text by removing non-alphanumeric characters (except spaces and punctuation)
with open('../data/text8', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace non-alphanumeric characters with spaces
cleaned_content = re.sub(r'[^a-zA-Z0-9\s.,!?;:()\-\'\"]', ' ', content)

# Split cleaned content into pseudo-sentences (insert newline every 100 words)
words = cleaned_content.split()
chunk_size = 100  # Adjust this value if needed
sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Write the modified content with newlines to the file
with open('../data/cleaned_text8.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sentences))


print("File cleaned and saved as 'cleaned_text8.txt'")
