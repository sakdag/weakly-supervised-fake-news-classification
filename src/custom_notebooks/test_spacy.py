import spacy

from src.weak_labeling.consts.spacy_consts import PUNCTUATION, SYMBOL, SPACE

nlp = spacy.load("en_core_web_sm")


if __name__ == "__main__":
    doc = nlp(
        'Apple is looking at "buying" U.K. : (startup) for $1 billion? Hey, you! ok.'
    )

    # for sentence in doc.sents:
    #     print(len(sentence))

    num_of_words = 0
    word_lengths = []
    for token in doc:
        if token.pos_ not in [PUNCTUATION, SYMBOL, SPACE]:
            num_of_words += 1
            word_lengths.append(len(token.text))
        # print(
        #     token.text,
        #     token.lemma_,
        #     token.pos_,
        #     token.tag_,
        #     token.dep_,
        #     token.shape_,
        #     token.is_alpha,
        #     token.is_stop,
        # )

    print(num_of_words)
    print(word_lengths)
