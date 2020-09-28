import os
import pandas as pd
import argparse


def main(experiments_paths):
	experiments = [x for x in os.listdir(experiments_paths) if os.path.isdir(os.path.join(experiments_paths, x))]

	lines = []
	header = None

	for experiment in experiments:
		experiment_path = os.path.join(experiments_paths, experiment)

		datasets_names = [x for x in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, x))]
		for dataset_name in datasets_names:
			df = pd.read_csv(os.path.join(experiment_path, dataset_name, 'overall', 'summary.csv'), header=[0,1], index_col=0)
			line_selector = ['-mean-of-means' in x for x in df.index]
			data = df.loc[line_selector, ('unweighted_area_under_roc', 'mean')]

			if header is None:
				header = ['experiment_name', 'dataset_name'] + data.index.tolist()

			lines += [[experiment, dataset_name] + data.values.tolist()]
	
	last = pd.DataFrame(lines, columns=header)			
	last.to_csv(os.path.join(experiments_paths, 'final_summary.csv'), index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Call this cript after postprocess.'
	)

	parser.add_argument('--experiments-paths', action='store', required=True, 
	help='Path to experiments folder. Works recursively '
	'(i.e. as long as .csv files are within the specified folder, it will work).')

	args = parser.parse_args()
	main(experiments_paths=args.experiments_paths)  