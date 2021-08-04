import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from .video_text_loader import get_raw_video, get_text
import os
import glob

def next_batch(data, batch_size):
    n_batch = int(np.ceil(data.shape[0] / batch_size))
    for i in range(n_batch):
        yield data[i * batch_size: min((i+1) * batch_size, data.shape[0])]

def extract_visual_features(args, task):
    # inputs_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
    input_frames = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, None, 3))
    
    ### load model pretrained on HowTo100M (S3D or I3D)
    if args.pretrain_model == 's3d':
        module = hub.Module("https://tfhub.dev/deepmind/mil-nce/s3d/1")
    elif args.pretrain_model == 'i3d':
        module = hub.Module("https://tfhub.dev/deepmind/mil-nce/i3d/1")
    
    vision_output = module(input_frames, signature='video', as_dict=True)
    video_embedding = vision_output['video_embedding']
    
    video_data_dir = os.path.join(args.data_dir, task, 'videos')
    video_list = [video_name for video_name in glob.glob(video_data_dir + '/*')]
    dst_data_dir = os.path.join(args.processed_dir, task, 'video_embd')
    os.makedirs(dst_data_dir, exist_ok=True)
    for video_path in video_list:
        fid = os.path.splitext(os.path.basename(video_path))[0]
        print(fid)
        video = get_raw_video(video_path, args.framerate, args.n_frames)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.tables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
        
            video_loader = next_batch(video, batch_size=args.pretrain_batch)
        
            output = np.zeros([0, video_embedding.shape[-1]])
            for video_batch in video_loader:
                output_batch = sess.run(video_embedding, feed_dict={input_frames: video_batch})
                output = np.concatenate([output, output_batch], axis=0)
        
            np.save(os.path.join(dst_data_dir, fid + '_video_embeddings.npy'), output)

def extract_textual_features(args, task):
    # inputs_words are just a list of sentences (i.e. ['the sky is blue', 'someone cutting an apple'])
    input_words = tf.compat.v1.placeholder(tf.string, shape=(None))
    
    ### load model pretrained on HowTo100M (S3D or I3D)
    if args.pretrain_model == 's3d':
        module = hub.Module("https://tfhub.dev/deepmind/mil-nce/s3d/1")
    elif args.pretrain_model == 'i3d':
        module = hub.Module("https://tfhub.dev/deepmind/mil-nce/i3d/1")
    
    text_output = module(input_words, signature='text', as_dict=True)
    text_embedding = text_output['text_embedding']
    
    text_data_dir = os.path.join(args.processed_dir, task, 'verb_phrases')
    text_list = [text_name for text_name in glob.glob(text_data_dir + '/*_verb_phrases.txt')]
    dst_data_dir = os.path.join(args.processed_dir, task, 'text_embd')
    os.makedirs(dst_data_dir, exist_ok=True)
    for text_path in text_list:
        fid = os.path.basename(text_path).replace('_verb_phrases.txt', '')
        text = get_text(text_path, pos=-1)
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.tables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
        
            output = sess.run(text_embedding, feed_dict={input_words:text})
            np.save(os.path.join(dst_data_dir, '{}_verb_phrases_embeddings.npy'.format(fid)), output)
