# EDNEL: Estimation of Dependency Networks Algorithm for Ensemble Learning

## Getting started

After you have generated a .jar for the project, run it by invoking the following command:

```
java -jar ednel.jar
```

This will bring a list of parameters for the program. An successful call to the code is

```
java -jar weka.jar --datasets_path keel_datasets_10fcv --datasets_names german --metadata_path metadata 
--variables_path resources/distributions --options_path resources/options.json 
--n_generations 100 --n_individuals 100 --n_samples 1 --learning_rate 0.5 --selection_share 0.5 --burn_in 100 
--thinning_factor 0 --early_stop_generations 10 --early_stop_tolerance 0.001 --log true --nearest_neighbor 3
```

## Interpreting data

### Setup

Create a new Anaconda environment:


```bash
conda create --name ednel python=3.7 --yes
conda activate ednel
conda install --file requirements.txt
conda install -c alubbock pygraphviz=1.5 graphviz=2.41
conda install -c rmg pydot=1.4.1
```
 
Add the path to graphviz ```<folder_to_anaconda_installation>Anaconda3/pkgs/graphviz-2.41-0/Scripts``` to your system 
path

### Generating predictive performance metrics metrics

---

If during the training process, flag `--log` was set to `true`, it will be possible to generate a report on the various
 predictive performance metrics available in Weka:

```bash
python src/main/python/postprocess.py --csv-path <path_to_run>
```

Example:

```bash
python src/main/python/postprocess.py --csv-path metadata/02-06-2020-09-00-18
```

### Generating structure graph and probabilities tables

The same can be done for generating a visualization of the structure of evolutionary algorithm during the training
process:

```bash
python src/main/python/plot_network.py --json-path <path_to_run>/dependency_network_structure.json
```

Example:

```bash
python src/main/python/plot_network.py 
--json-path "C:\Users\henry\Projects\trash\28-06-2020-14-14-49\dummygerman\sample_01_fold_01\dependency_network_structure.json"
```

### Generating map of characteristics

Another postprocess that can be done is to generate a map of the explored solution space, for a given run of EDNEL:

```bash
python src/main/python/characteristics_to_pca.py 
--csv-path "<path_to_run>\characteristics.csv"
```

Example:

```bash
python src/main/python/characteristics_to_pca.py 
--csv-path "D:\Users\henry\Projects\trash\28-06-2020-14-14-49\dummygerman\sample_01_fold_01\characteristics.csv"
```