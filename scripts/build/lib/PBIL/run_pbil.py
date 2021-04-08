from PBIL.pbil.model import PBIL
from PBIL.utils import read_datasets
from weka.core import jvm


def main():
    jvm.start(max_heap_size='4g')  # using 4GB of heap size for JVM

    train_data = read_datasets('keel_datasets_10fcv\\sonar\\sonar-10-1tra.arff')
    test_data = read_datasets('keel_datasets_10fcv\\sonar\\sonar-10-1tst.arff')

    pbil = PBIL(resources_path='resources', train_data=train_data, n_generations=2, n_individuals=10)
    overall, last = pbil.run(1)
    print(overall.predict(test_data))
    jvm.stop()


if __name__ == '__main__':
    main()
