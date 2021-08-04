import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--data_dir', type=str, default='data',
            help='the directory that stores store (video and subtitles)')
    parser.add_argument('--processed_dir', type=str, default='processed',
            help='the directory that saves processed data'
                 '(verb phrases, pretrained embeddings')
    parser.add_argument('--output_dir', type=str, default='output',
            help='the directory that saves outputs (models, predictions)')
    parser.add_argument('--conc_fname', type=str, default='preprocessing/narr_process/concrete.txt',
            help='the filename of the concreteness scores '
                 '(need to be specified if conc_threshold is larger than 0')
    parser.add_argument('--task', type=str, default='105222',
            help='the task')

    # narration process config
    parser.add_argument('--perform_punct', action='store_false',
            help='whether punctuate subtitles, default: true')
    parser.add_argument('--perform_coref', action='store_false',
            help='whether perform coreference resolution, default: true')
    parser.add_argument('--save_sents', action='store_true',
            help='whether save sentences, default: false')
    parser.add_argument('--ignore_stop', action='store_false',
            help='whether ignore phrases only containing stop words, '
                 'default: true')
    parser.add_argument('--conc_threshold', type=float, default=3,
            help='ignore phrases with concreteness score lower than threshold, '
                 'range: [0, 5]')

    # pretrained feature extraction config
    parser.add_argument('--pretrain_model', type=str, default='i3d',
        choices=['s3d', 'i3d'], help='pretrained model (s3d or i3d)')
    parser.add_argument('--pretrain_batch', type=int, default=16,
            help='batch size for extracting pretrained visual embeddings')
    parser.add_argument('--framerate', type=int, default=10,
            help='framerate for extracting pretrained visual embeddings')
    parser.add_argument('--n_frames', type=int, default=32,
            help='number of frames for each segment '
                 '(default segment length: 32 / 10 = 3.2s')

    # feature learning module config
    parser.add_argument('--feat_dim', type=int, default=512,
            help='feature dimension')

    # SOPL config
    parser.add_argument('--n_iter', type=int, default=5,
            help='number of iterations in sopl')
    parser.add_argument('--smooth_param', type=float, default=0.001,
            help='smooth parameter (beta)')
    parser.add_argument('--center_init', type=str, default='kmeans++', choices=['random', 'kmeans++'],
            help='prototype initialization')
    parser.add_argument('--time_weight', type=float, default=1,
            help='timestamp weight (gamma)')
    parser.add_argument('--K', type=int, default=0,
            help='the number of visual clusters, '
                 'use groundtruth number of key-steps if K is 0')
    parser.add_argument('--K_text_bg', type=int, default=10,
            help='the number of extra clusters for background in text')

    # DWSA config
    parser.add_argument('--lr', type=float, default=0.0005,
            help='learning rate')
    parser.add_argument('--wd', type=float, default=0.02,
            help='weight decay')
    parser.add_argument('--delta_e', type=float, default=1,
            help='the cost for alignment with empty slots (delta_e in paper),'
                 'range: [0, 2]')
    parser.add_argument('--batch_size', type=int, default=30,
            help='batch size')
    parser.add_argument('--max_epoch', type=int, default=50,
            help='the number of training epochs')

    # Segmentation config
    parser.add_argument('--bg_ratio', type=float, default=0.4,
            help='the ratio of background, range: [0, 1]')
    parser.add_argument('--test_epoch', type=int, default=30,
            help='the epoch of the model to test')

    args = parser.parse_args()
    return args
