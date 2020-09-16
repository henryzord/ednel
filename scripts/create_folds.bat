REM command for generating 10 folds of a dataset on windows.

FOR /L %G IN (1,1,10) DO java -classpath "D:\Users\Henry\weka-3-9-3\weka.jar" weka.filters.unsupervised.instance.RemoveFolds -S 0 -N 10 -F %G -i "C:\Users\henry\Projects\ednel\keel_datasets_10fcv\hcvegypt\hcvegypt.arff" -o hcvegypt-10-%Gtst.arff
FOR /L %G IN (1,1,10) DO java -classpath "D:\Users\Henry\weka-3-9-3\weka.jar" weka.filters.unsupervised.instance.RemoveFolds -S 0 -N 10 -F %G -V -i "C:\Users\henry\Projects\ednel\keel_datasets_10fcv\hcvegypt\hcvegypt.arff" -o hcvegypt-10-%Gtra.arff