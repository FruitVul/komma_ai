from komma_ai.pre_processing import get_sentence_corpus

test_corpus = "Das hier ist der erste Satz. Das hier ist der zweite Satz."


def test_get_sentence_corpus():
    corpus = get_sentence_corpus([test_corpus])
    print(corpus)