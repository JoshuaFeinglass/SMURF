import json
from smurf.eval import preprocess,smurf_eval_captions
from smurf.system_analysis import smurf_system_analysis

ref_file = 'data/karpathy_ref.json'
cand_file = 'data/karpathy_cand.json'
result_file = 'results/smurf_scores.json'

#load and preprocess example caption set
ref_list = json.loads(open(ref_file, 'r').read())
cand_list = json.loads(open(cand_file, 'r').read())
cands = [preprocess(cap['caption']) for cap in cand_list]
ref_dict = {cap['image_id']:[] for cap in ref_list}
for i,cap in enumerate(ref_list):
    if len(ref_dict[cap['image_id']])<5:
        ref_dict[cap['image_id']].append(preprocess(cap['caption']))

refs = [ref_dict[cap['image_id']] for cap in cand_list]

#perform caption-level analysis of example caption set
meta_scorer = smurf_eval_captions(refs, cands, fuse=True)
scores = meta_scorer.evaluate()
with open(result_file, 'w') as outfile:
    json.dump(scores, outfile)

#perform system-level analysis from Figure 1 of SMURF paper
plot_colors = ['saddlebrown', 'royalblue', 'crimson', 'lawngreen']
standardization_file = 'smurf/standardize_estimates.txt'
plot_file = 'results/system_plot.png'
analysis = smurf_system_analysis(in_file='results/smurf_scores.json')
analysis.load_standardized_scores(estimates_file=standardization_file)
analysis.generate_plot(plot_colors,out_file=plot_file)
model_penalties = analysis.compute_grammar_penalities()
for num,total_penalty in enumerate(model_penalties):
	print('Model %i Penalty: %f'%(num+1,total_penalty))
