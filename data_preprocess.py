import os
import glob
from args import get_args
from preprocessing.narr_process.subtitle_process import subtitle_process
from preprocessing.narr_process.compute_concrete_score import write_conc_scores
from preprocessing.extract_features.extract_embeddings import extract_visual_features, extract_textual_features

def data_preprocess(args, task):
    sub_data_dir = os.path.join(args.data_dir, task, 'subtitles')
    srt_fname_list = glob.glob(sub_data_dir + '/*.srt')
    dst_data_dir = os.path.join(args.processed_dir, task, 'verb_phrases')
    tmp_dir = os.path.join(args.processed_dir, task, 'tmp')
    
    ### Narration Processing
    vocabulary = set()
    for srt_fname in srt_fname_list:
        print(srt_fname)
        all_phrases = subtitle_process(srt_fname, dst_data_dir, tmp_dir, args.perform_punct, args.perform_coref, args.save_sents)
        vocabulary.update(all_phrases)

    # Compute concretenss score
    vocab_fname = os.path.join(args.processed_dir, task, 'vocab_concrete.txt')
    write_conc_scores(args.conc_fname, vocab_fname, vocabulary, args.ignore_stop)
    
    ## Pretrained Features Extraction
    extract_visual_features(args, task)
    extract_textual_features(args, task)


if __name__ == '__main__':
    args = get_args()
    if args.task == 'all':
        task_list = sorted(os.listdir(args.data_dir))
    else:
        task_list = [args.task]

    for task in task_list:
        data_preprocess(args, task)
