import json
import codecs
import os

from komma_ai.pre_processing import get_sentence_corpus,replace_abbreviations


def main():
    """
    Creates the sentence corpus dataset for later use, some preprocessing to fix sentence bugs.
    - Year Fix
    - Abbreviations
    (see pre_processing)
    """

    package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    europsarl_path = os.path.join(package_path, r"data\EUROPSARL\de")

    files = os.listdir(europsarl_path)

    europsarl_content = []

    for file in files:
        with codecs.open(os.path.join(europsarl_path, file), "r", 'utf-8') as f:
            europsarl_content_ = f.readlines()
        europsarl_content += [x.strip() for x in europsarl_content_]

    recipes_path = os.path.join(package_path, r"data\GERMAN_RECIPES\recipes.json")

    with open(recipes_path) as json_file:
        recipes_json = json.load(json_file)

    recipe_texts = []

    for recipe in recipes_json:
        instruction = recipe["Instructions"]
        instruction = replace_abbreviations(instruction)
        recipe_texts.append(instruction.replace(".", ". "))

    three_path = os.path.join(package_path, r"data\3MILLSENTENCES\deu_news_2015_3M-sentences.txt")

    with codecs.open(three_path, "r", 'utf-8') as f:
        sentences = f.readlines()
        processed_sentences = []
        for sentence in sentences[0:]:
            if len(sentence) > 8:
                split_sentence = sentence.split(";")[-1]
                split_sentence = split_sentence.rstrip()
                processed_sentences.append(split_sentence)

    print("Europsarl Sentences:",len(europsarl_content))
    print("Recipe Sentences:", len(recipe_texts))
    print("3MIL Sentences:", len(processed_sentences))
    text_corpus = europsarl_content + recipe_texts + processed_sentences
    print("Final N Sentences:", len(text_corpus))

    sentence_corpus = get_sentence_corpus(text_corpus)

    with open(os.path.join(package_path, "data\sentence_corpus\sentence_corpus.txt"), mode='w', encoding='utf-8') as f:
        for sentence in sentence_corpus:
            f.write("%s\n" % sentence)


if __name__ == "__main__":
    main()


