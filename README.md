# VisualNarrationProceL-CVPR21

## Overview 

This repository contains the implementation of [Learning To Segment Actions From Visual and Language Instructions via Differentiable Weak Sequence Alignment](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Learning_To_Segment_Actions_From_Visual_and_Language_Instructions_via_CVPR_2021_paper.pdf) published in CVPR 2021.

> We address the problem of unsupervised action segmentation and feature learning in instructional videos using both visual and language instructions. Our key contributions include Soft Ordered Prototype Learning (SOPL) module and Differentiable Weak Sequence Alignment (DWSA) loss.


## Prerequisites
To install all the dependency packages, please run: 
> pip install -r requirements.txt

## Data Preparation

## Data Preprocessing
### Narration Processing
Our narration processing module can extract verb phrases from SRT-Format subtitles. Other formats, i.e. VTT or TTML, can be converted via `./preprocessing/narr_process/format_converter.py`  

Our module mainly includes the following parts:  
1. Punctuate the subtitles [1]  (this takes a long time)
2. Perform coreference resolution
3. Extract verb phrases from sentences
4. Compute the concreteness score of verb phrases [2]  


### Pretrained Feature Extraction

## Training

## Evaluation

## Citation
If you find the project helpful, we would appreciate if you cite the work:

> @article{Shen-VisualNarrationProceL:CVPR21,  
>           author = {Y.~Shen and L.~Wang and E.~Elhamifar},  
>           title = {Learning to Segment Actions from Visual and Language Instructions via Differentiable Weak Sequence Alignment},  
>          journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},  
>           year = {2021}}

## Reference
[1] We use punctuator from https://github.com/ottokart/punctuator2  
[2] The concreteness rating list is from https://github.com/ArtsEngine/concreteness  
[3] We use the I3D/S3D models pretrained on Howto100M dataset from https://www.di.ens.fr/willow/research/mil-nce/  
[4] The code for DWSA loss is adapted from pytorch-softdtw https://github.com/Sleepwalking/pytorch-softdtw  
