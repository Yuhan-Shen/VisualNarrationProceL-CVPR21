# VisualNarrationProceL-CVPR21
By Yuhan Shen, PhD student at Northeastern University
Email: shen.yuh@northeastern.edu

This repository contains the code for the narration processing module in our CVPR 2021 Paper 
"Learning to Segment Actions from Visual and Language Instructions via Differentiable Weak Sequence Alignment"

Required Packages:
  neuralcoref
  spacy (python -m spacy download en) to download model
  pysrt
  theano (for punctuator)

Pretrained Model (for punctuator):
  Demo-Europarl-EN.pcl
  https://drive.google.com/drive/folders/0B7BsN5f2F1fZQnFsbzJ3TWxxMms?resourcekey=0-6yhuY9FOeITBBWWNdyG2aw
  Please download this model from the URL and copy it to the directory punctuator

We provide Python implementation of the narration processing module, which includes:
1. Punctuate the subtitles 
   (if you need it, please download the pretrained model and copy it to the directory punctuator)
2. Perform coreference resolution
3. Extract verb phrases from sentences
4. Compute the concreteness score of verb phrases

If you need to run the narration processing module for SRT-Format video subtitles, 
please see the Python file subtitle_process.py
The video subtitles with other formats (i.e. VTT or TTML) can be converted via format_converter.py.

If you need to run the narration processing module for Plain-TXT files,
please see the Python file text_process.py

If you need to compute the concreteness score,
please see the Python file compute_concrete_score.py

Reference:
https://github.com/ottokart/punctuator2
