# Prepare Datasets



Here provide processing codes for video-language datasets to turn their original data formats into the format used in this repo, as shown in [data/105222](https://github.com/tiangeluo/VisualNarrationProceL-CVPR21/tree/main/data/105222). We currently support ProceL, and would include CrossTask later.



1. Go to ProceL (homepage)[https://www.khoury.northeastern.edu/home/eelhami/procel.htm] and download the annotation file (.zip)[https://drive.google.com/file/d/1b8PoZlYeNMP3PieJ3_80KkeibaCmmljS/view].
2. Unzip `ProceL_dataset.zip` and delete the `cpr_0041` line in `/ProceL_dataset/perform_cpr/readme.txt`. The dataset has error for that item. It did not include videos/annotations for `cpr_0041` line.
3. Move `extract_labels.py` under `/ProceL_dataset` and run it. `extract_labels.py` will extract required information from `data.mat` in each category and save in corresponding directories. 
4. Download videos and move to corresponding directories `/videos/`. 
