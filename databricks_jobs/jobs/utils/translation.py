from transformers import pipeline
from typing import List
model_checkpoint = "Helsinki-NLP/opus-mt-fr-en"
translator = pipeline("translation", model=model_checkpoint)


def truncate_long_words(text: str, limit=20) -> str:
    """
    if a word of more than limit characters, it is truncated.
    it's probably not a word, but something like "/" * 100
    """
    return ' '.join([w[:limit] for w in text.split(' ')])


def split_too_long_sentence(sentence: str, max_nb_words: int = 200) -> str:
    """
    it can happen that a text has no '.'
    In this case, this function arbitrarily cuts it in chunks of max_nb_words words
    """
    words = sentence.split(' ')
    splited_sentence_as_list = [((' '.join(words[x:x + max_nb_words])).strip()) for x in
                                range(0, len(words), max_nb_words)]
    return splited_sentence_as_list


def split_paragraph_in_sentences(paragraph: str, max_words: int) -> List[str]:
    """
    splits a text in sentences ( words originally separated by '.')
    if a sentence wontains too many words, it arbitrarily cuts it in chunks of max_words words
    """
    sentences = []
    for sentence in [t.strip() for t in paragraph.split('.') if t != '']:
        split_sentence = split_too_long_sentence(sentence, max_words)
        for s in split_sentence:
            sentences.append(s)
    return sentences


def split_in_grouped_sentences(text: str, max_words: int = 200):
    """
    this function splits string in group of sentences (ie : group of words initially separated by '.' in string) and
    return a list of concatenated sentences, with each group having a max of max_words words (strings separated by ' ')
    """

    text = truncate_long_words(text)
    sentences = split_paragraph_in_sentences(text, max_words)

    if not sentences:
        return ['']

    grouped_sentences = [sentences.pop(0).strip() + '. ']
    grouped_sentence_index = 0

    while sentences:

        words_in_current_sentence = grouped_sentences[grouped_sentence_index].split(' ')
        words_in_next_sentence = sentences[0].split(' ')

        if len(words_in_current_sentence) + len(words_in_next_sentence) <= max_words:
            grouped_sentences[grouped_sentence_index] += sentences.pop(0).strip() + '. '

        else:
            grouped_sentences.append(sentences.pop(0) + '. ')
            grouped_sentence_index += 1

    return grouped_sentences


def fr_en_translate(paragraph: str):
    """
    The model can't translate long text. That's why we have to split the text before translation.
    """
    paragraph = paragraph.strip()  # avoids the 'No, no, no, no, etc' effect (which is the translation of " ")
    batches = split_in_grouped_sentences(paragraph)
    translations = []

    for batch in batches:
        batch_translation = translator(batch)[0]['translation_text'] if batch else ''
        translations.append(batch_translation)
    translation = ' '.join(translations)

    return translation
