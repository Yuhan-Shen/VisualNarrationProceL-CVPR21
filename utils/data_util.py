import numpy as np
import csv
import os
import itertools
import glob

def multimodal_dataloader(video_embd_list, text_embd_list, batch_size, max_epoch):
    N = len(video_embd_list)
    n_batch = int(np.ceil(N / batch_size))
    i = 0
    n = 0
    while n < max_epoch:
        video_data = np.concatenate(video_embd_list[i * batch_size: min((i+1) * batch_size, N)], axis=0)
        text_data = np.concatenate(text_embd_list[i * batch_size: min((i+1) * batch_size, N)], axis=0)
        yield video_data, text_data
        i += 1
        if i >= n_batch:
            i = 0
            n += 1

def load_mapping(fname):
    """
      Load the dictionary of index to label from mapping file
    """
    mapping = dict()
    with open(fname, 'r') as f:
        for line in f:
            splits = line.strip().split()
            mapping[int(splits[0])] = splits[1]
    return mapping

def get_annotation_by_fid(fid, annot_dir):
    annot = []
    annot_fname = os.path.join(annot_dir, fid+'.csv')
    if not os.path.exists(annot_fname):
        print('{} not exists'.format(annot_fname))
        return None

    with open(annot_fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in reader:
            step_id = int(row[0])
            begin_time = float(row[1])
            end_time = float(row[2])
            annot.append([step_id, begin_time, end_time])
    return annot

def get_annotations(fid_list, annot_dir):
    annotations = {}
    for fid in fid_list:
        annot = get_annotation_by_fid(fid, annot_dir)
        annotations[fid] = annot
    return annotations

def load_video_data_with_time(video_dir, fid_list, gamma=1):
    """
      Load pretrained video embeddings with time-stamp in the last dimension
      Args:
        video_dir: directory that saves video embeddings
        fid_list: the list of file IDs to load
        gamma: time-stamp weight
      Returns:
        video_embd_list: the list of video embeddings, with time-stamp in the last dimension
    """
    video_embd_list = []
    
    for fid in fid_list:
        video_path = os.path.join(video_dir, fid+'_video_embeddings.npy')
        video_embd = np.load(video_path)
        
        video_embd /= np.sqrt(np.sum(video_embd**2, axis=-1, keepdims=True))+1e-10
        times = (np.arange(video_embd.shape[0])[:, np.newaxis] + 0.5) / video_embd.shape[0]
        video_embd = np.concatenate([video_embd, np.sqrt(gamma) * times], axis=-1)
        video_embd_list.append(video_embd)

    return video_embd_list
        
def read_vocab_concretes(vocab_fname):
    vocab_concretes = {}
    with open(vocab_fname, 'r') as f:
        for line in f:
            line_split = line.strip().split('\t')
            phrase = line_split[0]
            score = float(line_split[1])
            vocab_concretes[phrase] = score
    return vocab_concretes

def load_text_data_with_time(text_dir, embd_dir, fid_list, gamma=1, conc_threshold=3, vocab_fname=''):
    """
      Load pretrained textual embeddings with time-stamp in the last dimension
      Args:
        text_dir: directory that saves verb phrases
        embd_dir: directory that saves verb phrase embeddings
        fid_list: the list of file IDs to load
        gamma: time-stamp weight
        conc_threshold: threshold for concreteness (ignore phrases with lower scores)
        vocab_fname: filename that saves all phrases and concreteness, required if conc_threshold > 0
      Returns:
        embd_list: the list of textual embeddings, with time-stamp in the last dimension
        phrase_list: the list of verb phrases
    """
    if conc_threshold > 0:
        vocab_concretes = read_vocab_concretes(vocab_fname)
        kept_vocab = [k for k in vocab_concretes if vocab_concretes[k] >= conc_threshold]
    
    id_list = []
    embd_list = []
    phrase_list = []
    for fid in fid_list:
        text_fname = os.path.join(text_dir, fid+'_verb_phrases.txt')
        data_tlist = [l.strip().split('\t') for l in open(text_fname, 'r').readlines()]
        times_1 = np.array([float(d[0]) for d in data_tlist])
        times_2 = np.array([float(d[1]) for d in data_tlist])
        phrase_i = np.array([d[2] for d in data_tlist])
        
        embd_i = np.load(glob.glob(os.path.join(embd_dir, fid+'*.npy'))[0])
        embd_i /= np.sqrt(np.sum(embd_i**2, axis=-1, keepdims=True)) + 1e-10
        
        if conc_threshold > 0:
            keep_i = np.array([phrase in kept_vocab for phrase in phrase_i], dtype=np.bool)
        else:
            keep_i = np.ones(len(phrase_i), dtype=np.bool)

        if times_1.shape[0] > 0:
            frac_1 = times_1 / times_2[-1]
            frac_2 = times_2 / times_2[-1]
        else:
            frac_1 = times_1
            frac_2 = times_2

        frac_middle = np.array([frac_1, frac_2]).mean(axis=0, keepdims=True).T
        embd_i = np.concatenate([embd_i, gamma*frac_middle], axis=-1)

        phrase_i = phrase_i[keep_i]
        embd_i = embd_i[keep_i]
        embd_list.append(embd_i)
        phrase_list.append(phrase_i)
    return embd_list, phrase_list

def annot_to_framelabel(annot, dur, fps):
    """
       Get framewise labels
    """
    T = int(np.ceil(dur * fps))
    label = np.ones([T], dtype=np.int32) * -1
    for step_annot in annot:
        step, begin_time, end_time = step_annot
        begin_frame = int(np.floor(begin_time * fps))
        end_frame = int(np.ceil(end_time * fps))
        if end_frame > T:
            # actions start from 0, -1 is background 
            label[begin_frame:] = step - 1
            continue
        label[begin_frame:end_frame] = step - 1
    return label

def get_label_list(annot_dir, fid_list, dur_list, fps_list):
    """
       Load labels from annotations
    """
    annotations = get_annotations(fid_list, annot_dir)
    label_list = []
    for i, fid in enumerate(fid_list):
        label = annot_to_framelabel(annotations[fid], dur_list[i], fps_list[i])
        label_list.append(label)
    return label_list

def get_frame_pred(pred, dur, fps, t_segment=3.2):
    """
       convert segment-wise predictions to frame-wise
    """
    T = int(np.ceil(dur * fps))
    frame_pred = np.ones([T], dtype=np.int32) * -1
    group_pred = [(k, len(list(g))) for (k, g) in itertools.groupby(pred)]
    pos = 0
    for (k, l) in group_pred:
        begin_time, end_time = t_segment * pos, t_segment * (pos+l)
        pos += l
        if k == -1:
            continue
        begin_frame = int(np.floor(begin_time * fps))
        end_frame = min(int(np.ceil(end_time * fps)), T)
        frame_pred[begin_frame:end_frame] = k
    return frame_pred

def get_frame_pred_list(pred_list, video_dur_list, video_fps_list, t_segment=3.2):
    new_pred_list = []
    for i, pred in enumerate(pred_list):
        new_pred = get_frame_pred(pred, video_dur_list[i], video_fps_list[i], t_segment)
        new_pred_list.append(new_pred)
    return new_pred_list
