with open("./data/text8","r",encoding="utf-8") as f :
    text=f.read().lower()

words=text.split()
vocab=set(words)
print("Vocabulary length: ",len(vocab))