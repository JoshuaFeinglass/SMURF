#!/bin/bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install nltk
pip install transformers
pip install sentencepiece
pip install shapely
pip install sklearn
pip install pandas
pip install matplotlib
DIR="$(cd "$(dirname "$0")" && pwd)"
python3 $DIR/download_nltk_info.py
