import json
import pathlib
import re
from typing import List, Tuple

import nltk
from keras.preprocessing.text import Tokenizer
from pymystem3 import Mystem
from tqdm import tqdm

nltk.download("stopwords")
from nltk.corpus import stopwords

CORRECT_LEMMATIZE = False
CHECK_STOPWORDS = False

input_first_file_name = "Волшебник изумрудного города.txt"
input_second_file_name = "Удивительный Волшебник Из Страны Оз.txt"


def count_bag(
    sentences_of_first_text: List[str], sentences_of_second_text: List[str]
) -> Tuple:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_of_first_text + sentences_of_second_text)
    sequences_of_first_text = tokenizer.texts_to_sequences(sentences_of_first_text)
    sequences_of_second_text = tokenizer.texts_to_sequences(sentences_of_second_text)
    flat_sequences_of_first_text = [
        item for sub_sequences in sequences_of_first_text for item in sub_sequences
    ]
    flat_sequences_of_second_text = [
        item for sub_sequences in sequences_of_second_text for item in sub_sequences
    ]
    word_index = tokenizer.word_index
    bow = {}
    for key in word_index:
        counters = [
            flat_sequences_of_first_text.count(word_index[key]),
            flat_sequences_of_second_text.count(word_index[key]),
        ]
        bow[key] = [sum(counters), *counters]
    return bow, word_index


punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~–"


def has_punctuation(token: str) -> bool:
    for s in token.strip():
        if s in punctuation:
            return True
    return False


russian_stopwords = stopwords.words("russian")


def is_stopword(token: str) -> bool:
    return CHECK_STOPWORDS and token in russian_stopwords


mystem = Mystem()
word_regex = re.compile(r"([\wЁёА-я]+)")


def clear_sentence(text):
    if CORRECT_LEMMATIZE:
        tokens = mystem.lemmatize(text.strip().lower())
        tokens = filter(
            lambda token: token.strip()
            and not is_stopword(token)
            and not has_punctuation(token),
            tokens,
        )
    else:
        tokens = filter(
            lambda token: not is_stopword(token),
            word_regex.findall(text.strip().lower()),
        )
    text = " ".join(tokens)
    return text


def get_output_file_name():
    return f"{input_first_file_name.strip('.txt')}_{input_second_file_name.strip('.txt')}_{'lemmatized' if CORRECT_LEMMATIZE else 'not_lemmatized'}_{'without_stopwords' if CHECK_STOPWORDS else 'with_stopwords'}.json"


def main():
    input_first_file_path = pathlib.Path.cwd() / "input" / input_first_file_name
    input_second_file_path = pathlib.Path.cwd() / "input" / input_second_file_name
    with open(
        str(input_first_file_path), "r", encoding="UTF-8"
    ) as input_first_file, open(
        str(input_second_file_path), "r", encoding="UTF-8"
    ) as input_second_file:
        first_raw_text = input_first_file.readlines()
        second_raw_text = input_second_file.readlines()
        first_text = [
            clear_sentence(sentence)
            for sentence in tqdm(first_raw_text, unit="line", desc="prepare_sentences")
            if sentence.strip()
        ]
        second_text = [
            clear_sentence(sentence)
            for sentence in tqdm(second_raw_text, unit="line", desc="prepare_sentences")
            if sentence.strip()
        ]
        bag, word_index = count_bag(first_text, second_text)
        print(f"Найдено {len(word_index)} уникальных слов.")
        output_file_name = get_output_file_name()
        output_file_path = pathlib.Path.cwd() / "output" / output_file_name
        output_file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(str(output_file_path), "w+", encoding="UTF-8") as output_file:
            json.dump(bag, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
