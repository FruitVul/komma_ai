import pandas as pd
import os
from nltk.tokenize import sent_tokenize


def get_sentence_corpus(text_file):
    """Returns the sentences of a text_file
    Arguments:
        text_file(list): Loaded Text.

    Returns:
        sentences(list): All found texts.
    """
    sentences = []
    for text in text_file:
        if "<" not in text:
            text_sentences = replace_abbreviations(text)
            text_sentences = fix_year_error(text_sentences)
            text_sentences = sent_tokenize(text_sentences)
            for sentence in text_sentences:
                sentence = sentence.replace("PUNKT",". ")
                sentences.append(sentence.replace("daß", "dass").replace("muß", "muss"))
    return sentences


def replace_abbreviations(text):
    """Replaces abbreviations by fully written words
    Arguments:
        text(string): text.

    Returns:
        sentence(string): Sentence without abbreviations.
    """
    if "Ca. " in text:
        text = text.replace("Ca.", "Zirka")
    if "ca." in text:
        text = text.replace("ca.", "zirka")
    if "bzw." in text:
        text = text.replace("bzw.", "beziehungsweise")
    if "Min." in text:
        text = text.replace("Min.", "Minuten")
    if "min." in text:
        text = text.replace("min.", "Minuten")
    if "z.B." in text:
        text = text.replace("z.B.", "zum Beispiel")
    if "z. B." in text:
        text = text.replace("z. B.", "zum Beispiel")
    if "Evtl." in text:
        text = text.replace("Evtl.", "Eventuell")


    return text


def fix_year_error(text):
    """Fixes an error where Date. Year (f.e. 17. Januar) is split up into two sentences
        Arguments:
            text(string): text.

        Returns:
            text(string): text with PUNKT as identifier.
    """

    months = ["Januar","Februar","März","April",
              "Mai","Juni","Juli","August",
              "September","Oktober","November","Dezember"]
    for i,char in enumerate(text):
        if char.isdigit():
            if i + 14 < len(text):
                if ". " in text[i+1:i+3]:
                    for month in months:
                        if month in text[i:i+14]:
                            text = text[0:i+1] + "PUNKT" + text[i+2:]

    return text


def remove_special_chars(text, remove_chars="<.:-()\"{}\´!?>@%;&[]+~#_|$€=/"):
    """Removes special characters
        Arguments:
            text(string): text.
            remove_chars(string): string with all chars to be replaced.

        Returns:
            text(string): text without the special chars.
    """

    for special_char in remove_chars:
        text = text.replace(special_char, "")
    return text


def tokenize(sentence):
    split_sentence = split_up_sentence(sentence)
    tokens = []
    for idx,split in enumerate(split_sentence):
        token = get_token(idx,split,len(split_sentence))
        tokens.append(token)
    return tokens

def split_up_sentence(sentence):
    sentence_parts = sentence.split(" ")
    split_sentence = []
    for i, part in enumerate(sentence_parts):
        if "," in part:
            split_sentence.append(part.replace(",", ""))
            split_sentence.append(",")
        else:
            split_sentence.append(part)
            if i < len(sentence_parts) - 1:
                split_sentence.append(" ")

    return split_sentence


def get_token(idx,split,sentence_len):
    if check_if_number_split(split):
        return "DIGIT"
    if len(split) == 1 and split != "," and split != " ":
        return "CHAR"
    if split == " ":
        return "SPACE"
    if idx == 0:
        return "SOS"
    if idx == sentence_len-1:
        return "EOS"
    if "," in split:
        return "KOMMA"
    else:
        return split


def check_if_number_split(split):
    n_number = 1
    n_str = 1
    for char in split:
        if char.isdigit():
            n_number += 1
        else:
            n_str += 1

    ratio = n_number / (n_str+n_number)

    return True if ratio > 0.5 else False
