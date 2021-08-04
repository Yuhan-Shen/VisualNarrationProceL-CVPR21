# VisualNarrationProceL-CVPR21

## Overview 

This repository contains the implementation of [Learning To Segment Actions From Visual and Language Instructions via Differentiable Weak Sequence Alignment](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Learning_To_Segment_Actions_From_Visual_and_Language_Instructions_via_CVPR_2021_paper.pdf) published in CVPR 2021.

> We address the problem of unsupervised action segmentation and feature learning in instructional videos using both visual and language instructions. Our key contributions include Soft Ordered Prototype Learning (SOPL) module and Differentiable Weak Sequence Alignment (DWSA) loss.


## Prerequisites
To install all the dependency packages, please run: 
```
pip install -r requirements.txt
```
To download the model for spacy, please run:
```
python -m spacy download en
```
`ffmpeg` is required for loading videos. If you do not have it installed, you may run
```
conda install ffmpeg
```
or 
```
apt-get install ffmpeg
```
or any alternatives.

Info: It works on Python version: 3.7.11, Cuda version: 10.1

## Data Preparation
### Dataset
We include two video samples under the directory `./data/105222` from the task `make kimchi fried rice` in CrossTask dataset to show the required hierachy. You may run the following commands using these videos.

For full dataset, please refer to [CrossTask Dataset](https://github.com/DmZhukov/CrossTask).

If you want to use ProceL dataset, please refer to [ProceL](https://www.khoury.northeastern.edu/home/eelhami/procel.htm).

### Pretrained Model
  We use the pretrained model `Demo-Europarl-EN.pcl` as punctuator. If you want to perform punctuation to subtitles, please download this model from the [URL](https://drive.google.com/drive/folders/0B7BsN5f2F1fZQnFsbzJ3TWxxMms?resourcekey=0-6yhuY9FOeITBBWWNdyG2aw) (only Demo-Europarl-EN.pcl is needed) and put it to the folder `preprocessing/narr_process/punctuator/`.


## Usage
After downloading data, the model can be run by:
  ```
  python main.py --data_dir <data_dir> --task <task>
  ```
where <data_dir> is the directory that you save the data and <task> is the task you want to use, <task> can also be `all` if you want to run the model for all tasks in one run. If you want to use the sample videos, simply run `python main.py`.  
This command will run the whole process of **data preprocessing**, **training**, and **segmentation**. 
  
In the end, it will print the segmentation results as:
 
> Task: 105222, Precision: 18.60%, Recall: 38.35%, F1-score: 25.05%, MoF: 41.77%, MoF-bg: 38.35%
 
Note: The data preprocessing module takes a long time, but it only needs to be executed once. The processed video embeddings and textual embeddings will be stored in the folder `args.processed/args.task`. You may also execute data preprocessing, training, and segmentation separately as follows.
  
## Data Preprocessing
The data preprocessing module includes two parts: narration processing (extract verb phrases from subtitles) and extract pretrained features. It can be run by:
  ```
  python data_preprocess.py --data_dir <data_dir> --task <task>
  ```
 The narration processing module can be time-consuming. If you want to skip punctuation, please run
   ```
  python data_preprocess.py --data_dir <data_dir> --task <task> --perform_punct
  ```
Similarly, if you want to skip coreference resolution, please add `--perform_coref`. You can also skip the step of computing concreteness scores by setting concreteness threshold to be zero (using `--conc_threshold 0`).
  
Note: If you are interested in the details of data preprocessing, please see the rest of this section. Otherwise, you may move to the **Training** section.
  
### Narration Processing
Our narration processing module can extract verb phrases from SRT-Format subtitles. Other formats, i.e. VTT or TTML, can be converted via `./preprocessing/narr_process/format_converter.py`  

Our module mainly includes the following parts:  
1. Punctuate the subtitles [1]  (this takes a long time)
2. Perform coreference resolution
3. Extract verb phrases from sentences
4. Compute the concreteness score of verb phrases [2]  

### Pretrained Feature Extraction
We use the pretrained model [3] to extract visual embeddings of video segments and textual embeddings of verb phrases in a joint embedding space.
The videos should be stored in `args.data_dir/args.task/videos` and the verb phrases should be stored in `args.processed_dir/args.task/verb_phrases`. The extracted features will be stored in `args.processed_dir/args.task/video_embd` and `args.processed_dir/args.task/text_embd`.
  
You may choose either pretrained S3D or I3D model. This can be set by adding argument `--pretrain_model s3d` or `--pretrain_model i3d`.
## Training
The training module (multimodal feature learning using DWSA) can be run by:
  ```
  python train.py --data_dir <data_dir> --task <task>
  ```
 You may change the hyparameters by arguments, such as learning rate (`--lr`), weight decay (`--wd`), batch size (`--batch_size`), max epoch (`--max_epoch`).  
  Other hyparameters can be set in a similar manner: timestamp weight (`--time_weight`), smoothing parameter (`--smooth_param`), empty alignment cost (`--delta_e`), and so on.
  
  
## Segmentation
After training, we can load the trained model to get new features, and apply clustering to the new features to get the segmentations. Please run:
The training module (multimodal feature learning using DWSA) can be run by:
  ```
  python segmentation.py --data_dir <data_dir> --task <task> --test_epoch <test_epoch> --bg_ratio <bg_ratio>
  ```  
 where <test_epoch> is the epoch of the model you want to test on, <bg_ratio> is the ratio of background in your segmentations (default: 0.4).

## Citation
If you find the project helpful, we would appreciate if you cite the work:

```
@article{Shen-VisualNarrationProceL:CVPR21,  
         author = {Y.~Shen and L.~Wang and E.~Elhamifar},  
         title = {Learning to Segment Actions from Visual and Language Instructions via Differentiable Weak Sequence Alignment},  
         journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},  
         year = {2021}}
```

## Reference
[1] We use punctuator from https://github.com/ottokart/punctuator2  
[2] The concreteness rating list is from https://github.com/ArtsEngine/concreteness  
[3] We use the I3D/S3D models pretrained on Howto100M dataset from https://www.di.ens.fr/willow/research/mil-nce/  
[4] The code for DWSA loss is adapted from pytorch-softdtw https://github.com/Sleepwalking/pytorch-softdtw  

## Contact
shen [dot] yuh [at] northeastern [dot] edu
