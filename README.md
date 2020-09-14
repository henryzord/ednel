# EDNEL: Estimation of Dependency Networks Algorithm for Ensemble Learning

## Getting started

### Installing

#### On Windows:

1. Download Java JDK SE version **8u261** from Oracle site: 
https://www.oracle.com/br/java/technologies/javase/javase-jdk8-downloads.html (note: you'll need to have an Oracle 
account to do so, to accept their software use agreement)
2. Install using the Graphical Installer
	Note: installing JDK on windows should have already installed the Java Runtime Environment (JRE). If not, 
		download version **8u241** here: https://www.java.com/en/download/win10.jsp
3. Test whether these two programs are in PATH variable:
	```bash
    java -version
    javac -version
    ```

If any of the programs was not found, add their installation path to the PATH system variable (the installation path 
is the one selected in steps 1 and 2)

Note: the code uses JDK version 8, independent of the downloaded version.

#### On Ubuntu (tested version 18.04):

You still have to download JDK from Oracle. On Ubuntu this is a little more complicated because you have to use a third
party tool to download it. Follow these steps:

1. Create an Oracle account
2. Download version 14.0.2 from this page: https://www.oracle.com/java/technologies/javase-jdk14-downloads.html
Choose **Linux Compressed Archive**
3. Unpack it wherever it seems fit in your machine, e.g. in your home directory: `tar -xf jdk-14.0.2_linux-x64_bin.tar.gz`
4. A new folder of name `jdk-14.0.2` was created in this folder. Add this path to your PATH variable:
    a. Open the .bashrc file in your HOME directory, e.g. `nano .bashrc`
    b. Pre-pend the path to `jdk-14.0.2/bin` in your PATH variable: `export PATH=:/home/<your_user>/jdk-14.0.2/bin:$PATH`
    c. You will have to either close and open the terminal or give a `source .bashrc` for modifications to have 
    immediate effect
    c. Check whether everything is working:
    ```bash
   which java
   which javac
    ```
   It should point to your new installation path, as opposed to the system-wide java/javac.

Install Weka version `3.9.3`
Download it from Source Forge: https://sourceforge.net/projects/weka/files/weka-3-9/3.9.3/


#### TODO add instructions from building code from command line

### Using

After you have generated a .jar for the project, run it by invoking the following command:

```
java -jar ednel.jar
```

This will bring a list of parameters for the program. An successful call to the code is

```
java -jar ednel.jar --datasets_path keel_datasets_10fcv --datasets_names vehicle,pima,wisconsin,flare,australian,german,bupa,contraceptive --metadata_path metadata --n_generations 200 --n_individuals 200 --n_samples 1 --learning_rate 0.7 --selection_share 0.5 --burn_in 100 --thinning_factor 0 --early_stop_generations 200 --early_stop_tolerance 0.001 --log --max_parents 1 --delay_structure_learning 1 --n_jobs 5 
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

If during the training process, flag `--log` was set, it will be possible to generate a report on the various
 predictive performance metrics available in Weka:

```bash
python scripts/postprocess.py --csv-path <path_to_run>
```

Example:

```bash
python scripts/postprocess.py --csv-path metadata/02-06-2020-09-00-18
```

### Generating dashboard of training metadata

The same can be done for generating a visualization of the structure of evolutionary algorithm during training, as well
as the explored solution space, probability tables, and population metrics:

```bash
python scripts/plot_network.py --experiment-path <path_to_run>
```

Example:

```bash
python scripts/plot_network.py --experiment-path "C:\Users\henry\Projects\trash\28-06-2020-14-14-49\dummygerman\sample_01_fold_01"
```

## Testing compatibility of datasets

Use class `TestDatasets` to test whether EDNEL will successfuly run on a dataset:

```bash
java -classpath ednel.jar ednel.TestDatasets --datasets_path <datasets_path> --datasets_names null --metadata_path null --n_generations 2 --n_individuals 25 --n_samples 1 --learning_rate 0.7 --selection_share 0.5 --burn_in 2 --thinning_factor 0 --early_stop_generations 200 --early_stop_tolerance 0.001 --max_parents 0 --delay_structure_learning 1 --n_jobs 1 --timeout 300
```
