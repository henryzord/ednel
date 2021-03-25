import os
import argparse
import json
from collections import Counter
import pandas as pd


def unpack(d: dict, experiment_path: str):
	"""
	Transforms a dictionary of hyper-parameters and their counts (per dataset) in a table, where only variable
	hyper-parameters are discriminated (e.g. if all datasets use the same value for one hyper-parameter, this 
	hyper-parameter will not be shown.)
	"""
	columns = set()

	raw_params = []

	datasets = []
	for param in d.keys():
		set_values = set()
		datasets = d[param].keys()
		for dataset in d[param].keys():
			for k in d[param][dataset].keys():
				set_values = set_values.union({k})
		if (len(set_values) > 1) and ('dataset' not in param.lower()):
			raw_params += [param]
			for k in d[param][dataset].keys():
				columns = columns.union({param + '_' + k})

	df = pd.DataFrame(data=0, columns=sorted(columns), index=datasets, dtype=int)

	for param in raw_params:
		for dataset in d[param].keys():
			for value, count in d[param][dataset].items():
				df.loc[dataset, param + '_' + value] = count

	df.to_csv(os.path.join(experiment_path, 'ultra_parameters.csv'), index=True)
	print(df)


def main(experiment_path: str):
	variable_parameters = dict()

	experiments = [x for x in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, x))]
	for experiment in experiments:
		dataset = [x for x in os.listdir(os.path.join(experiment_path, experiment)) if os.path.isdir(os.path.join(experiment_path, experiment, x))][0]
		
		param_files = [x for x in os.listdir(os.path.join(experiment_path, experiment, dataset)) if 'parameters.json' in x]

		d = dict()
		
		for param_file in param_files:
			with open(os.path.join(experiment_path, experiment, dataset, param_file), 'r') as ff:
				jf = json.load(ff)
				for k, v in jf.items():
					try:
						d[k] += [v]
					except KeyError:
						d[k] = [v]
		
		if len(d) > 0:
			# print('%s:' % dataset)			
			for k, v in d.items():
				cc = Counter(v)
				if k not in variable_parameters:
					variable_parameters[k] = dict()
				variable_parameters[k][dataset] = dict()
				for name, count in cc.items():
					variable_parameters[k][dataset][name] = count

	unpack(variable_parameters, experiment_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='After running nestedcv experiments, use this script to interpret hyper-parameters of tested methods.'
	)

	parser.add_argument(
		'--experiment-path', action='store', required=True,
		help='The path where several experiments are contained.'
	)

	args = parser.parse_args()
	main(experiment_path=args.experiment_path)
