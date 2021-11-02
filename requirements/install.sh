#!/bin/bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install nltk
pip install transformers
pip install sentencepiece
pip install shapely
pip install sklearn
pip install pandas
python3 download_nltk_info.py
