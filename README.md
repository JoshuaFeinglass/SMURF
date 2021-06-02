# SeMantic and linguistic UndeRstanding Fusion (SMURF)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic caption evaluation metric described in the paper [SMURF: SeMantic and linguistic UndeRstanding Fusion for Caption Evaluation via Typicality Analysis](FILL SOON) (ACL 2021).

### Overview
SMURF is an automatic caption evaluation metric that combines a novel semantic evaluation algorithm (SPARCS) and novel fluency evaluation algorithms (SPURTS and MIMA) for both caption-level and system-level analysis. These evaluations were developed to be generalizable as a result demonstrate a high correlation with human judgment across many relevant datasets. See paper for more details.

### Requirements
- torch>=1.0.0
- numpy
- nltk>=3.5.0
- pandas>=1.0.1
- matplotlib
- transformers>=3.0.0

### Usage

./smurf_example.py provides working examples of the following functions:

#### Caption-Level Scoring
Returns a dictionary with scores for semantic similarity between reference captions and candidate captions (SPARCS), style/diction quality of candidate text (SPURTS), grammar outlier penalty of candidate text (MIMA), and the fusion of these scores (SMURF).

#### System-Level Analysis
After reading in and standardizing caption-level scores, generates a plot that can be used to give an overall evaluation of captioner performances along with relevant system-level scores (intersection with reference captioner and total grammar outlier penalties) for each captioner.
![](./results/system_plot.png "system_analysis")
The number of captioners you are comparing should be specified by when instantiating a smurf_system_analysis object. In order to generate the plot correctly, the captions of each candidate captioner (C1, C2, ..., CN) should be organized in the following format: 
[C1 output image 1, C2 output image 1, ... CN output image 1, C1 output image 2, C2 output image 2, CN output image 2].
The C1 captioner should be the ground truth.

### Author:
Joshua Feinglass (https://scholar.google.com/citations?user=V2h3z7oAAAAJ&hl=en)

If you find this repo useful, please cite:
```
@inproceedings{feinglass2021smurf,
  title={SMURF: SeMantic and linguistic UndeRstanding Fusion for Caption Evaluation via Typicality Analysis},
  author={Joshua Feinglass and Yezhou Yang},
  booktitle={},
  year={2021},
  url={FILL SOON}
}
```