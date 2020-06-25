# eda.EDNEL: Estimation of Dependency Networks Algorithm for Ensemble Learning

## Getting started

After you have generated a .jar for the project, run it by invoking the following command:

```
java -jar ednel.jar
```

This will bring a list of parameters for the program. An successful call to the code is

```
java -jar weka.jar --datasets_path keel_datasets_10fcv --datasets_names german --metadata_path metadata --variables_path resources/distributions --options_path resources/options.json --n_generations 100 --n_individuals 100 --n_samples 1 --learning_rate 0.5 --selection_share 0.5 --burn_in 100 --thinning_factor 0 --early_stop_generations 10 --early_stop_tolerance 0.001 --log true --nearest_neighbor 3
```

## Crunching data

First, create a new Anaconda environment:


```bash
conda create --name ednel python=3.7 --yes
conda activate ednel
conda install --file requirements.txt
conda install -c alubbock pygraphviz=1.5 graphviz=2.41
conda install -c rmg pydot=1.4.1
```

Last but not least, add 
And just add the path to graphviz ```<folder_to_anaconda_installation>Anaconda3/pkgs/graphviz-2.41-0/Scripts``` to your system path

---

Once finished, and if `--log` is set to `true`, result files will be stored in the specified metadata path. Crunch the data using the following script/command:

```bash
python src/main/python/postprocess.py --csv-path <metadata_path>
```

Example:

```bash
python src/main/python/postprocess.py --csv-path metadata/02-06-2020-09-00-18
```