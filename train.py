import numpy as np
import torch
import os
import torch.optim as optim
from args import get_args

from utils.data_util import load_video_data_with_time, load_text_data_with_time, multimodal_dataloader

from model import MultiModal_SOPL
from losses.dwsa_loss import DWSA_Loss

np.random.seed(123)
#torch.manual_seed(0)

def run_train(args, task):
    video_dir = os.path.join(args.processed_dir, task, 'video_embd')
    text_dir = os.path.join(args.processed_dir, task, 'verb_phrases')
    text_embd_dir = os.path.join(args.processed_dir, task, 'text_embd')
    
    video_fid_list = sorted([f.replace('_video_embeddings.npy', '')  
        for f in os.listdir(video_dir) if f.endswith('.npy')])
    text_fid_list = sorted([f.replace('_verb_phrases.txt', '')  
        for f in os.listdir(text_dir) if f.endswith('_verb_phrases.txt')])
    fid_list = [fid for fid in video_fid_list if fid in text_fid_list]

    #annotations = get_annotations(fid_list, annot_dir)

    #dur_list, fps_list, label_list = get_info_label(fid_list, info_dict, annotations)
    
    print(len(fid_list))
    video_embd_list = load_video_data_with_time(video_dir, fid_list, gamma=np.sqrt(args.time_weight))
    video_lens = [video_embd.shape[0] for video_embd in video_embd_list]
    print(video_lens)

    
    #K = int(labels.max()) + 1 if args.K == 0 else args.K


    vocab_fname = os.path.join(args.processed_dir, task, 'vocab_concrete.txt')
    text_embd_list, phrases_list = load_text_data_with_time(text_dir, text_embd_dir, fid_list, gamma=np.sqrt(args.time_weight), conc_threshold=args.conc_threshold, vocab_fname=vocab_fname)

    #K_text_bg = 10
    #learning_rate = 0.0005
    #wd = 0.02


    # TO DO: K, K_text_bg,
    K = 6
    model = MultiModal_SOPL(K, K + args.K_text_bg, args.feat_dim, args.n_iter, args.smooth_param)
                    
    model.cuda()
    criterion = DWSA_Loss(args.smooth_param, threshold=args.delta_e, softmax='row')
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    data_loader = multimodal_dataloader(video_embd_list, text_embd_list, args.batch_size, max_epoch=args.max_epoch)
    
    n_iter = 0
    n_epoch = 0
    N = len(video_embd_list)
    n_batch = int(np.ceil(N / args.batch_size))

    for v_data, t_data in data_loader:
        data = torch.from_numpy(v_data).double().cuda()
        text_data = torch.from_numpy(t_data).double().cuda()
        x_centers, y_centers, x, y = model(data[:, :-1], text_data[:, :-1], data[:, -1:], text_data[:, -1:])
        loss = criterion(x_centers, y_centers)

        if n_iter % n_batch == 0:
            if (n_epoch + 1) % 10 == 0:
                PATH = '{}/{}/models/model_{}.pth'.format(args.output_dir, task, n_epoch + 1)

                dirname = os.path.dirname(PATH)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                model.save(PATH)
            n_epoch += 1
        n_iter += 1
    
        print('epoch {}, iter {}, loss: {}'.format(n_epoch, n_iter, loss.cpu().item()))
        model.zero_grad()
        loss.backward()
        optimizer.step()




if __name__ == '__main__':

    args = get_args()
    if args.task == 'all':
        task_list = sorted(os.listdir(args.data_dir))
    else:
        task_list = [args.task]

    for task in task_list:
        run_train(args, task)
