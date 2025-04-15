import requests

url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
text = requests.get(url).text

with open("ml_course_en/lecture 17. Transformer/jane_austen/pride_and_prejudice.txt", "w", encoding="utf-8") as f:
    f.write(text)
