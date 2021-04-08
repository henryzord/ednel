"""
A script for analysing results of experiments.
Procedurally removes datasets from an experiment sample until there is a statistically significant difference between
the EDA version and Random Forest.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import copy
from collections import Counter
import warnings


def main():
    df = pd.read_csv('meta_dataset.csv', index_col=0)

    baselines = ['random_forest']
    algorithms = set(df.columns) - set(baselines)

    res_raw = []

    plain_list = []

    warnings.filterwarnings('error')

    for algorithm in algorithms:
        for baseline in baselines:
            remove_set = list()
            sub = df[algorithm] - df[baseline]  # type: pd.Series
            sub = sub.sort_values(ascending=True)
            sorted_datasets = copy.deepcopy(sub.index.tolist())
            statistic, p_value = wilcoxon(sub)
            last_removed = -1
            while p_value >= 0.05 and len(remove_set) < len(sorted_datasets):
                last_removed += 1
                remove_set += [sorted_datasets[last_removed]]
                sub.pop(sorted_datasets[last_removed])
                try:
                    statistic, p_value = wilcoxon(sub)
                except UserWarning:
                    break

            res_raw += [[algorithm, baseline, p_value, len(remove_set), remove_set]]
            plain_list.extend(remove_set)

    counter = Counter(plain_list)
    print(counter)

    res = pd.DataFrame(res_raw, columns=['algorithm', 'baseline', 'p value', 'n_removed', 'removed datasets'])
    res.to_csv('removed.csv', index=False)
    print(res)


if __name__ == '__main__':
    main()
