import json
from smurf.eval import preprocess,smurf_eval_captions
import numpy as np
fh = open('data/karpathy_test_set.json','r')
data_in = json.load(fh)
id_dict = {img['filename']:[sent['raw'] for sent in img['sentences']]
           for img in data_in['images'] if img['split'] == 'test'}
refs = []
cands = []
for img in id_dict.keys():
    anns = id_dict[img]
    ref = []
    for i,ann in enumerate(anns):
        if i==0:
            cands.append(preprocess(ann))
        else:
            ref.append(preprocess(ann))
    refs.append(ref)

meta_scorer = smurf_eval_captions(refs, cands, fuse=False)
scores = meta_scorer.evaluate()

SPARCS_param = (np.mean(scores['SPARCS']),np.std(scores['SPARCS']))
SPURTS_param = (np.mean(scores['SPURTS']),np.std(scores['SPURTS']))
MIMA_param = (np.mean(scores['MIMA']),np.std(scores['MIMA']))

out_file = open('smurf/standardize_estimates.txt', "w")
out_file.write("SPARCS,%f,%f\n" % SPARCS_param)
out_file.write("SPURTS,%f,%f\n" % SPURTS_param)
out_file.write("MIMA,%f,%f\n" % MIMA_param)
out_file.close()