import re
from collections import Counter


class TextPreprocessor:

    def __init__(self):
        self.word_to_index = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.index_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def build_vocabulary(self, texts):

        word_counter = Counter()

        for text in texts:

            text = self.clean_text(text)

            words = text.split()

            for word in words:
                word_counter[word] += 1

        for word in sorted(word_counter.keys()):

            if word not in self.word_to_index:

                index = len(self.word_to_index)

                self.word_to_index[word] = index
                self.index_to_word[index] = word

    def encode_text(self, text):

        text = self.clean_text(text)

        words = text.split()

        encoded = []

        for word in words:

            if word in self.word_to_index:
                encoded.append(self.word_to_index[word])
            else:
                encoded.append(self.word_to_index["<UNK>"])

        return encoded
