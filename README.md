# eda.EDNEL: Estimation of Dependency Networks Algorithm for Ensemble Learning

## Command line example

java -c eda.Main --datasets_path C:\Users\henry\Projects\ednel\keel_datasets_10fcv --datasets_names german --metadata_path C:\Users\henry\Projects\trash --variables_path C:\Users\henry\Projects\ednel\resources\distributions --options_path C:\Users\henry\Projects\ednel\resources\options.json --n_generations 100 --n_individuals 100 --n_samples 1 --learning_rate 0.7 --selection_share 0.5 --burn_in 100 --thinning_factor 0 --early_stop_generations 10 --early_stop_tolerance 0.001 --log true --nearest_neighbor 3