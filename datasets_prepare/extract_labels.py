import scipy.io
import numpy as np
from IPython import embed
import os
import pandas as pd
import datetime

def format_td(seconds, digits=3):
    isec, fsec = divmod(round(seconds*10**digits), 10**digits)
    return f'{datetime.timedelta(seconds=isec)}.{fsec:0{digits}.0f}'

mother_path = '/Users/tiange/Downloads/ProceL_dataset'
task_names = []
for _,dirs,_ in os.walk(mother_path):
    task_names.append(dirs)
task_names = task_names[0]

for n in task_names:
    path = os.path.join(mother_path,n)
    mat = scipy.io.loadmat(os.path.join(path,'data.mat'))
    key_len = mat['grammar'].shape[0]
    with open(os.path.join(path,'mapping.txt'),'w') as f:    
        for i in range(key_len):
            f.writelines('%d %s\n'%(i+1,mat['grammar'][i][0][0].replace(' ','_')))

    readme_txt = pd.read_csv(os.path.join(path,'readme.txt'), delimiter = "\t")
    file_names = readme_txt[readme_txt.keys()[1]].to_list()
    file_names.insert(0,readme_txt.keys()[1])

    #annotations
    anno_dir = os.path.join(path,'annotations')
    if not os.path.exists(anno_dir):
        os.mkdir(anno_dir)
    for i,name in enumerate(file_names):
        with open(os.path.join(anno_dir,name+'.csv'),'w') as f:
            for j in range(key_len):
                arr = mat['key_steps_time'][i][0][j][0]
                if arr.shape[0] == 0:
                    continue
                for k in range(arr.shape[0]):
                    f.writelines('%d,%.2f,%.2f\n'%(j+1,arr[k,0],arr[k,1]))

    #subtitles
    sub_dir = os.path.join(path,'subtitles')
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    for i,name in enumerate(file_names):
        with open(os.path.join(sub_dir,name+'.srt'),'w') as f:
            counter = 1
            for j in range(mat['caption'][i][0].shape[0]):
                if mat['caption'][i][0][j][0].shape[0] == 0:
                    continue
                f.writelines('%d\n'%(counter))
                counter += 1
                f.writelines('%s --> %s\n'%('0'+format_td(mat['caption_time'][i][0][j][0][0][0]).replace('.',','),'0'+format_td(mat['caption_time'][i][0][j][0][0][1]).replace('.',',')))
                f.writelines('%s\n'%mat['caption'][i][0][j][0][0][1:])
                f.writelines('\n')
