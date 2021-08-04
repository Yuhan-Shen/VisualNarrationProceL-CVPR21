import spacy
from .stop_words import ENGLISH_STOP_WORDS


def compute_conc_score(verb_phrase, conc_dict, ignore_stop=True):
    """
      compute the concreteness score for verb phrase, which is the highest concreteness score for component words
      if ignore_stop is True, the score of verb phrase that only contains stop words is zero.
    """
    nlp = spacy.load('en')
    max_score = 0
    is_stop = True
    for token in nlp(verb_phrase):
        conc_score = max(conc_dict.get(token.lemma_.lower(), 0), conc_dict.get(token.text.lower(), 0))
        if not (token.text in conc_dict or token.lemma_ in conc_dict):
            # if the word is not in vocabulary, concrete score for nouns or proper nouns is 3.5, otherwise 0
            if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                conc_score = max(conc_score, 3.5)
        if token.lemma_.lower() not in ENGLISH_STOP_WORDS and token.text.lower() not in ENGLISH_STOP_WORDS:
            is_stop = False
        max_score = max(max_score, conc_score)
    if ignore_stop and is_stop:
        max_score = 0
    return max_score

def get_conc_dict(fname):
    """
     Load the concreteness score dict from the file
    """
    conc_dict = {}
    with open(fname, 'r') as f:
        f.readline()
        for line in f.readlines():
            word = line.strip().split('\t')[0]
            conc = line.strip().split('\t')[2]
            conc = float(conc)
            conc_dict[word] = conc
    return conc_dict

def write_conc_scores(conc_fname, dst_fname, phrases, ignore_stop):
    conc_dict = get_conc_dict(conc_fname)
    with open(dst_fname, 'w') as f:
        for phrase in phrases:
            score = compute_conc_score(phrase, conc_dict, ignore_stop)
            f.write('{}\t{}\n'.format(phrase, score))

if __name__ == '__main__':
    conc_dict = get_conc_dict('concrete.txt')
    verb_phrase = 'change your car tire'
    score = compute_conc_score(verb_phrase, conc_dict, ignore_stop=True)
    print(score)
