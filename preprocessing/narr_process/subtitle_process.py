import spacy
import neuralcoref
import os
import numpy as np
import pysrt

def read_srt_subtitle(fname):
    """
      Read the captions and times from SRT-format subtitles;
      Other formats can be converted first
    """
    subs = pysrt.open(fname)
    captions = [sub.text for sub in subs]
    times = [[sub.start.ordinal / 1000, sub.end.ordinal / 1000] for sub in subs]
    return captions, times

def punctuate_captions(captions, fid, tmp_dir):
    """
      Punctuate the captions
      This function will generate two temp files with extension ".tmp1" and ".tmp2"
      tmp1 is the whole original captions; tmp2 is the punctuated captions
    """
    input_fname = os.path.join(tmp_dir, fid+'.tmp1')
    output_fname = os.path.join(tmp_dir, fid+'.tmp2')

    text = ' '.join(captions)
    with open(input_fname, 'w') as f:
        f.write(text)
    ### run punctuator
    os.system('cat {} | python preprocessing/narr_process/punctuator/punctuator.py preprocessing/narr_process/punctuator/Demo-Europarl-EN.pcl {}'.format(input_fname, output_fname))
    with open(output_fname, 'r') as f:
        text = f.read()
    return text

def remove_extra_space(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

def captions_to_sents(captions, times, text, perform_coref=True):
    """
    split the captions into sentences, also estimate the rough time of each sentence
    """
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    ## Get the number of tokens (excluding punctations) for each line
    num_tokens = []
    token_lines = []
    for line in captions:
        line = remove_extra_space(line)
        doc_line = nlp(line)
        token_line = []
        num_puncts = 0
        for token in doc_line:
            if token.is_punct:
                num_puncts += 1
            else:
                token_line.append(token.text)
        token_lines.append(token_line)
        length = len(doc_line) - num_puncts
        num_tokens.append(length)

    ## Get the number of accumulated tokens (excluding punctations) up to each line
    num_acc_tokens = [[sum(num_tokens[:i]), sum(num_tokens[:i+1])-1] for i in range(len(num_tokens))]
    num_acc_tokens = np.array(num_acc_tokens)

    text = remove_extra_space(text)
    doc = nlp(text)

    sents = [] # the list of sentences
    sents_time = [] # the list of estimated times for each sentence
    total_num_puncts = 0 # the total number of punctuations
    for j, sent in enumerate(doc.sents):
        sent_text = [token.string for token in sent]
        num_puncts = 0  # count the number of puncts in each sentence
        for token in sent:
            if token.is_punct:
                num_puncts += 1
            if perform_coref and token.dep_ == "dobj" and token.lemma_ == '-PRON-' and token._.in_coref:
                coref = token._.coref_clusters[0].main.root
                coref_text = coref.text
                #coref_lemma = coref.lemma_
                token_i_in_sent = token.i - sent.start
                sent_text[token_i_in_sent] = sent_text[token_i_in_sent].replace(token.text, coref_text)
                
        sents.append(''.join(sent_text))

        # the index of the start and end of the sentence (excluding puncts)
        idx_start, idx_end = sent.start-total_num_puncts, sent.end-1-total_num_puncts-num_puncts
        total_num_puncts += num_puncts
        # find the index of lines in caption
        idx_capt_start = np.where(num_acc_tokens[:, 0]<=idx_start)[0][-1]
        idx_capt_end = np.where(num_acc_tokens[:, 1]>=idx_end)[0][0]
        # estimate the start and end time of each sentence 
        # (the duration of each caption line is equally assigned to each character)
        ratio_start = sum([len(t) for t in token_lines[idx_capt_start][:idx_start-num_acc_tokens[idx_capt_start, 0]]]
                ) / sum([len(t) for t in token_lines[idx_capt_start]])
        start_time = (1 - ratio_start) * times[idx_capt_start][0] + ratio_start * times[idx_capt_start][1]
        ratio_end = sum([len(t) for t in token_lines[idx_capt_end][:idx_end-num_acc_tokens[idx_capt_end, 0]+1]]
                ) / sum([len(t) for t in token_lines[idx_capt_end]])
        end_time = (1 - ratio_end) * times[idx_capt_end][0] + ratio_end * times[idx_capt_end][1]
        sents_time.append([start_time, end_time])

    return sents, sents_time

def write_sents_with_time(sents, sents_time, fid, data_dir):
    assert len(sents) == len(sents_time)

    with open(os.path.join(data_dir, fid+'_sents.txt'), 'w') as f:
        for i, sent in enumerate(sents):
            f.write('{:.2f}\t{:.2f}\t{}\n'.format(sents_time[i][0], sents_time[i][1], sent))

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
            for child in prep_children:
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

def extract_verb_phrase_from_sents(sents, sents_time, fid, data_dir):
    all_phrases = set()
    fname = os.path.join(data_dir, '{}_verb_phrases.txt'.format(fid))
    with open(fname, 'w') as f:
        for j, sent in enumerate(sents):
            verb_phrases_list = extract_verb_phrase_from_sent(sent)
            all_phrases.update(set(verb_phrases_list))
            for verb_phrase in verb_phrases_list:
                f.write('{:.3f}\t{:.3f}\t{}\n'.format(sents_time[j][0], sents_time[j][1], verb_phrase))
    return all_phrases

def load_sents_from_file(fname):
    sents = []
    sents_time = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line_split = line.strip().split('\t')
            start_time, end_time = float(line_split[-3]), float(line_split[-2])
            sent = line_split[-1]
            sents_time.append([start_time, end_time])
            sents.append(sent)
    return sents, sents_time


def subtitle_process(srt_fname, data_dir='processed', tmp_dir='tmp', perform_punct=True, perform_coref=True, save_sents=True):
    """
      srt_fname: filename for SRT-format video subtitles
      data_dir: the directory to save the extracted verb phrases and sentences (optional, if save_sents is True)
      tmp_dir: the directory to save some temporary files (including raw text and punctated text)
      perform_punct: whether perform punctuation
      perform_coref: whether perform coreference resolution
      save_sents: whether save sentences in data_dir
    """
    fid = os.path.splitext(os.path.basename(srt_fname))[0]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    captions, times = read_srt_subtitle(srt_fname)
    if perform_punct:
        text = punctuate_captions(captions, fid, tmp_dir)
    else:
        text = ' '.join(captions)
    sents, sents_time = captions_to_sents(captions, times, text, perform_coref)
    if save_sents:
        write_sents_with_time(sents, sents_time, fid, data_dir)
    all_phrases = extract_verb_phrase_from_sents(sents, sents_time, fid, data_dir)
    return all_phrases

if __name__ == '__main__':
    srt_fname = '87706_aik2x6p4JLw.srt'
    data_dir = 'processed'
    tmp_dir = 'tmp'
    subtitle_process(srt_fname, data_dir, tmp_dir, True, False)
