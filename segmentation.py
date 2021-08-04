import torch
import numpy as np
import os
from sklearn.cluster import KMeans
from model import MultiModal_Features
from args import get_args
from utils.data_util import load_video_data_with_time, get_label_list, get_frame_pred_list, load_mapping
from utils.eval_util import framewise_eval

def video_segmentation(x, K, bg_ratio):
    kmeans = KMeans(n_clusters=K, random_state=None).fit(x)
    preds = kmeans.labels_
    centers = kmeans.cluster_centers_
    preds = np.array([l for l in kmeans.labels_])
    for k in range(K):
        pos = np.where(preds==k)[0]
        dist = np.sum((x[pos] - centers[k])**2, axis=-1)
        tmp = preds[pos]
        # predict segments as background if the distance from cluster center is large
        if dist.shape[0] > 0:
            threshold = np.sort(dist)[int((1 - bg_ratio) * dist.shape[0]) - 1]
            tmp[dist > threshold] = -1
        preds[pos] = tmp
    return preds

def run_test(args, task):

    video_dir = os.path.join(args.processed_dir, task, 'video_embd')
    fid_list = sorted([f.replace('_video_embeddings.npy', '')  
        for f in os.listdir(video_dir) if f.endswith('.npy')])

    video_embd_list = load_video_data_with_time(video_dir, fid_list, gamma=np.sqrt(args.time_weight))
    video_lens = [video_embd.shape[0] for video_embd in video_embd_list]
    concat_embd = np.concatenate(video_embd_list, axis=0)

    if args.K == 0:
        # set K to be the groundtruth number of steps if args.K == 0
        map_fname = os.path.join(args.data_dir, task, 'mapping.txt')
        mapping = load_mapping(map_fname)
        K = len(mapping)  
    else:
        K = args.K

    data = torch.from_numpy(concat_embd).double().cuda()
    model = MultiModal_Features(args.feat_dim)
    model.cuda()

    PATH = '{}/{}/models/model_{}.pth'.format(args.output_dir, task, args.test_epoch)
    model.load(PATH)
    model.eval()
    x = model.extract_visual_features(data[:, :-1], data[:, -1:])
    x = x.detach().cpu().numpy()
    preds = video_segmentation(x, K, args.bg_ratio)

    pred_list = [preds[sum(video_lens[:i]):sum(video_lens[:i+1])] for i in range(len(video_lens))]

    pred_dir = os.path.join(args.output_dir, task, 'pred')
    os.makedirs(pred_dir, exist_ok=True)
    for fid, pred in zip(fid_list, pred_list):
        pred_fname = os.path.join(pred_dir, fid + '.txt')
        np.savetxt(pred_fname, pred, fmt='%i')

    ### the duration and fps of videos (can be given if known, otherwise will be estimated by n_frames and framerate)
    dur_list = [video_len * args.n_frames / args.framerate for video_len in video_lens]
    fps_list = [args.framerate for i in video_lens]

    pred_list = get_frame_pred_list(pred_list, dur_list, fps_list, t_segment=args.n_frames / args.framerate)
    annot_dir = os.path.join(args.data_dir, task, 'annotations')
    label_list = get_label_list(annot_dir, fid_list, dur_list, fps_list)
    metric = framewise_eval(pred_list, label_list)
    print('Task: {}, Precision: {:.2%}, Recall: {:.2%}, F1-score: {:.2%}, '
          'MoF: {:.2%}, MoF-bg: {:.2%}'.format(task, *metric))


if __name__ == '__main__':

    args = get_args()
    if args.task == 'all':
        task_list = sorted(os.listdir(args.data_dir))
    else:
        task_list = [args.task]

    for task in task_list:
        run_test(args, task)
