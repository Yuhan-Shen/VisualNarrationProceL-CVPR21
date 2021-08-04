import spacy
import neuralcoref
import os

def remove_extra_space(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

def extract_verb_phrase_from_sent(sent):
    nlp = spacy.load('en')
    doc = nlp(sent)
    noun_chunks = [chunk for chunk in doc.noun_chunks]
    verb_phrases_list = []
    for token in doc:
        ### find all words with `direct object` dependency
        if token.dep_ == "dobj":
            token_noun_chunk = [chunk for chunk in noun_chunks if token in chunk]
            # use the noun chunk if the token is in a noun chunk
            token_output = token.lemma_ if len(token_noun_chunk) == 0 else token_noun_chunk[0]
            # find the verb of the `direct object`
            verb = token.head.lemma_
            # search for particle within all children of the verb
            for child in token.head.children:
                if child.dep_ == 'prt':
                    verb = verb + ' ' + child.text
            has_prep_flag = False # whether the verb has associated prep obj
            # search for `prep+pobj` within all children of the verb
            prep_children = [child for child in token.head.children if child.dep_ == 'prep']
            for chid in prep_children:
                pobj_subchildren = [subchild for subchild in child.children if subchild.dep_ == 'pobj']
                for subchild in pobj_subchildren:
                    if not token.lemma_ == '-PRON-' and not subchild.lemma_ == '-PRON-':
                        subchild_noun_chunk = [chunk for chunk in noun_chunks if subchild in chunk]
                        subchild_output = subchild.lemma_ if len(subchild_noun_chunk) == 0 else subchild_noun_chunk[0]
                        # verb (prt) + dobj + prep + pobj
                        verb_phrase = '{} {} {} {}'.format(verb, token_output, child.lemma_, subchild_output)
                        verb_phrases_list.append(verb_phrase)
                        has_prep_flag = True
            # only keep `verb+dobj` if no prep+pobj (skip the phrases with pronoun objects)
            if not has_prep_flag and not token.lemma_ == '-PRON-':
                verb_phrase = '{} {}'.format(verb, token_output)
                verb_phrases_list.append(verb_phrase)
    return verb_phrases_list

def extract_verb_phrase_from_sents(sents, fid, data_dir):
    fname = os.path.join(data_dir, '{}_verb_phrases.txt'.format(fid))
    with open(fname, 'w') as f:
        for j, sent in enumerate(sents):
            verb_phrases_list = extract_verb_phrase_from_sent(sent)
            for verb_phrase in verb_phrases_list:
                f.write('{}\n'.format(verb_phrase))

def punctuate_text(txt_fname, fid, tmp_dir):
    output_fname = os.path.join(tmp_dir, fid+'.tmp2')
    os.system('cat {} | python punctuator/punctuator.py punctuator/Demo-Europarl-EN.pcl {}'.format(txt_fname, output_fname))
    with open(output_fname, 'r') as f:
        text = f.read()
    return text

def text_to_sents(text, perform_coref=True):
    """
      split text into sentences
      optional: perform coreference resolution
    """
    nlp = spacy.load('en')
    if perform_coref:
        neuralcoref.add_to_pipe(nlp)

    text = remove_extra_space(text)
    doc = nlp(text)

    sents = [] # the list of sentences
    for j, sent in enumerate(doc.sents):
        sent_text = [token.string for token in sent]
        if perform_coref:
            for token in sent:
                if  token.dep_ == "dobj" and token.lemma_ == '-PRON-' and token._.in_coref:
                    coref = token._.coref_clusters[0].main.root
                    coref_text = coref.text
                    #coref_lemma = coref.lemma_
                    token_i_in_sent = token.i - sent.start
                    sent_text[token_i_in_sent] = sent_text[token_i_in_sent].replace(token.text, coref_text)
                
        sents.append(''.join(sent_text))
    return sents

def write_sents(sents, fid, data_dir):
    with open(os.path.join(data_dir, fid+'_sents.txt'), 'w') as f:
        for sent in sents:
            f.write('{}\n'.format(sent))

def text_process(txt_fname, data_dir='processed', tmp_dir='tmp', perform_punct=True, perform_coref=True, save_sents=True):
    """
      txt_fname: filename for PLAIN-TEXT
      data_dir: the directory to save the extracted verb phrases and sentences (optional, if save_sents is True)
      tmp_dir: the directory to save some temporary files (including punctated text)
      perform_punct: whether perform punctuation
      perform_coref: whether perform coreference resolution
      save_sents: whether save sentences in data_dir
    """
    fid = txt_fname[:-4]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    if perform_punct:
        text = punctuate_text(txt_fname, fid, tmp_dir)
    else:
        with open(txt_fname, 'r') as f:
            text = f.read()
    sents = text_to_sents(text, perform_coref)
    if save_sents:
        write_sents(sents, fid, data_dir)

    extract_verb_phrase_from_sents(sents, fid, data_dir)


srt_fname = '87706_aik2x6p4JLw.txt'
data_dir = 'processed'
tmp_dir = 'tmp'
text_process(srt_fname, data_dir, tmp_dir, perform_punct=True, perform_coref=True, save_sents=True)
