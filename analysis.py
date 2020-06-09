import csv
import pathlib

import numpy as np
from scipy.spatial import distance

input_file_name = "Волшебник изумрудного города_Удивительный Волшебник Из Страны Оз_not_lemmatized_with_stopwords.csv"


def calc_pearson_correlation(values):
    return np.corrcoef(*list(zip(*values)))


def calc_cosine_correlation(values):
    return 1 - distance.cosine(*list(zip(*values)))


def main():
    input_file_path = pathlib.Path.cwd() / "output" / input_file_name
    with open(str(input_file_path), "r", newline="") as input_file:
        words_bag = list(csv.reader(input_file))[1:]
        text_info = {}
        for word_info in words_bag:
            text_info[word_info[0]] = [int(word_info[2]), int(word_info[3])]
        pearson_correlation = calc_pearson_correlation(text_info.values())[0][1]
        cosine_correlation = calc_cosine_correlation(text_info.values())
        print(f"Корреляция по Пирсону: {pearson_correlation}\nКосинусная корреляция: {cosine_correlation}")


if __name__ == "__main__":
    main()
