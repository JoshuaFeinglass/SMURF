# <img src="/data/smurf_pic.png" width="60"> SeMantic and linguistic UndeRstanding Fusion (SMURF)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic caption evaluation metric described in the paper "SMURF: SeMantic and linguistic UndeRstanding Fusion for Caption Evaluation via Typicality Analysis" (ACL 2021).

arXiv: https://arxiv.org/abs/2106.01444

ACL Anthology: https://aclanthology.org/2021.acl-long.175/

### Overview
SMURF is an automatic caption evaluation metric that combines a novel semantic evaluation algorithm (SPARCS) and novel fluency evaluation algorithms (SPURTS and MIMA) for both caption-level and system-level analysis. These evaluations were developed to be generalizable and as a result demonstrate a high correlation with human judgment across many relevant datasets. See paper for more details.

### Requirements
- python 3
- torch>=1.0.0
- numpy
- nltk>=3.5.0
- pandas>=1.0.1
- matplotlib
- transformers>=3.0.0

### Usage

./smurf_example.py provides working examples of the following functions:

#### Caption-Level Scoring
Returns a dictionary with scores for semantic similarity between reference captions and candidate captions (SPARCS), style/diction quality of candidate text (SPURTS), grammar outlier penalty of candidate text (MIMA), and the fusion of these scores (SMURF). Input sentences should be preprocessed before being fed into the smurf_eval_captions object as shown in the example. Evaluations with SPARCS require a list of reference sentences while evaluations with SPURTS and MIMA do not use reference sentences.

#### System-Level Analysis
After reading in and standardizing caption-level scores, generates a plot that can be used to give an overall evaluation of captioner performances along with relevant system-level scores (intersection with reference captioner and total grammar outlier penalties) for each captioner. An example of such a plot is shown below:
![](./results/system_plot.png "system_analysis")

The number of captioners you are comparing should be specified when instantiating a smurf_system_analysis object. In order to generate the plot correctly, the captions fed into the caption-level scoring for each candidate captioner (C1, C2,...) should be organized in the following format with the C1 captioner as the ground truth: 

[C1 image 1 output, C2 image 1 output,..., C1 image 2 output, C2 image 2 output,...].

### Author/Maintainer:
Joshua Feinglass (https://scholar.google.com/citations?user=V2h3z7oAAAAJ&hl=en)

If you find this repo useful, please cite:
```
@inproceedings{feinglass2021smurf,
  title={SMURF: SeMantic and linguistic UndeRstanding Fusion for Caption Evaluation via Typicality Analysis},
  author={Joshua Feinglass and Yezhou Yang},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2021},
  url={https://aclanthology.org/2021.acl-long.175/}
}
```