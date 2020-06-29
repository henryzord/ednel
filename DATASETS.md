# KEEL Datasets

All datasets have been collected from [KEEL](https://sci2s.ugr.es/keel/index.php). Some of them present missing values (expressed in the column _n_missing_ 
with the number of instances with missing values).

All datasets are in .arff format and separated in 20 files, a train and test set for each one of the iterations of a 10-fold stratified cross validation.

| name | instances | attributes | categorical | numeric | classes | missing | class distribution |
|-----:|------------:|-------------:|--------------:|----------:|----------:|----------:|:------------------:|
| abalone | 4174 | 8 | 1 | 7 | 28 | 0 | ![abalone](figures/abalone.png) |
| adult | 48842 | 14 | 8 | 6 | 2 | 7.41% | ![adult](figures/adult.png) |
| appendicitis | 106 | 7 | 0 | 7 | 2 | 0 | ![appendicitis](figures/appendicitis.png) |
| australian | 690 | 14 | 6 | 8 | 2 | 0 | ![australian](figures/australian.png) |
| balance | 625 | 4 | 0 | 4 | 3 | 0 | ![balance](figures/balance.png) |
| banana | 5300 | 2 | 0 | 2 | 2 | 0 | ![banana](figures/banana.png) |
| breast | 286 | 9 | 9 | 0 | 2 | 3.15% | ![breast](figures/breast.png) |
| bupa | 345 | 6 | 0 | 6 | 2 | 0 | ![bupa](figures/bupa.png) |
| car | 1728 | 6 | 6 | 0 | 4 | 0 | ![car](figures/car.png) |
| chess | 3196 | 36 | 36 | 0 | 2 | 0 | ![chess](figures/chess.png) |
| cleveland | 303 | 13 | 0 | 13 | 5 | 1.98% | ![cleveland](figures/cleveland.png) |
| coil2000 | 9822 | 85 | 0 | 85 | 2 | 0 | ![coil2000](figures/coil2000.png) |
| connect-4 | 67557 | 42 | 42 | 0 | 3 | 0 | ![connect-4](figures/connect-4.png) |
| contraceptive | 1473 | 9 | 0 | 9 | 3 | 0 | ![contraceptive](figures/contraceptive.png) |
| crx | 690 | 15 | 9 | 6 | 2 | 5.36% | ![crx](figures/crx.png) |
| dermatology | 366 | 34 | 0 | 34 | 6 | 2.19% | ![dermatology](figures/dermatology.png) |
| ecoli | 336 | 7 | 0 | 7 | 8 | 0 | ![ecoli](figures/ecoli.png) |
| fars | 100968 | 29 | 24 | 5 | 8 | 0 | ![fars](figures/fars.png) |
| flare | 1066 | 11 | 11 | 0 | 6 | 0 | ![flare](figures/flare.png) |
| german | 1000 | 20 | 13 | 7 | 2 | 0 | ![german](figures/german.png) |
| glass | 214 | 9 | 0 | 9 | 7 | 0 | ![glass](figures/glass.png) |
| haberman | 306 | 3 | 0 | 3 | 2 | 0 | ![haberman](figures/haberman.png) |
| hayes-roth | 160 | 4 | 0 | 4 | 3 | 0 | ![hayes-roth](figures/hayes-roth.png) |
| heart | 270 | 13 | 0 | 13 | 2 | 0 | ![heart](figures/heart.png) |
| ionosphere | 351 | 33 | 0 | 33 | 2 | 0 | ![ionosphere](figures/ionosphere.png) |
| iris | 150 | 4 | 0 | 4 | 3 | 0 | ![iris](figures/iris.png) |
| kddcup | 494020 | 41 | 15 | 26 | 23 | 0 | ![kddcup](figures/kddcup.png) |
| led7digit | 500 | 7 | 0 | 7 | 10 | 0 | ![led7digit](figures/led7digit.png) |
| letter | 20000 | 16 | 0 | 16 | 26 | 0 | ![letter](figures/letter.png) |
| lymphography | 148 | 18 | 15 | 3 | 4 | 0 | ![lymphography](figures/lymphography.png) |
| magic | 19020 | 10 | 0 | 10 | 2 | 0 | ![magic](figures/magic.png) |
| mammographic | 961 | 5 | 0 | 5 | 2 | 13.63% | ![mammographic](figures/mammographic.png) |
| monk-2 | 432 | 6 | 0 | 6 | 2 | 0 | ![monk-2](figures/monk-2.png) |
| movement_libras | 360 | 90 | 0 | 90 | 15 | 0 | ![movement_libras](figures/movement_libras.png) |
| newthyroid | 215 | 5 | 0 | 5 | 3 | 0 | ![newthyroid](figures/newthyroid.png) |
| optdigits | 5620 | 64 | 0 | 64 | 10 | 0 | ![optdigits](figures/optdigits.png) |
| page-blocks | 5472 | 10 | 0 | 10 | 5 | 0 | ![page-blocks](figures/page-blocks.png) |
| penbased | 10992 | 16 | 0 | 16 | 10 | 0 | ![penbased](figures/penbased.png) |
| phoneme | 5404 | 5 | 0 | 5 | 2 | 0 | ![phoneme](figures/phoneme.png) |
| pima | 768 | 8 | 0 | 8 | 2 | 0 | ![pima](figures/pima.png) |
| post-operative | 90 | 8 | 8 | 0 | 3 | 3.33% | ![post-operative](figures/post-operative.png) |
| ring | 7400 | 20 | 0 | 20 | 2 | 0 | ![ring](figures/ring.png) |
| saheart | 462 | 9 | 1 | 8 | 2 | 0 | ![saheart](figures/saheart.png) |
| satimage | 6435 | 36 | 0 | 36 | 7 | 0 | ![satimage](figures/satimage.png) |
| segment | 2310 | 19 | 0 | 19 | 7 | 0 | ![segment](figures/segment.png) |
| sonar | 208 | 60 | 0 | 60 | 2 | 0 | ![sonar](figures/sonar.png) |
| spambase | 4597 | 57 | 0 | 57 | 2 | 0 | ![spambase](figures/spambase.png) |
| spectfheart | 267 | 44 | 0 | 44 | 2 | 0 | ![spectfheart](figures/spectfheart.png) |
| splice | 3190 | 60 | 60 | 0 | 3 | 0 | ![splice](figures/splice.png) |
| tae | 151 | 5 | 0 | 5 | 3 | 0 | ![tae](figures/tae.png) |
| texture | 5500 | 40 | 0 | 40 | 11 | 0 | ![texture](figures/texture.png) |
| thyroid | 7200 | 21 | 0 | 21 | 3 | 0 | ![thyroid](figures/thyroid.png) |
| tic-tac-toe | 958 | 9 | 9 | 0 | 2 | 0 | ![tic-tac-toe](figures/tic-tac-toe.png) |
| titanic | 2201 | 3 | 0 | 3 | 2 | 0 | ![titanic](figures/titanic.png) |
| twonorm | 7400 | 20 | 0 | 20 | 2 | 0 | ![twonorm](figures/twonorm.png) |
| vehicle | 846 | 18 | 0 | 18 | 4 | 0 | ![vehicle](figures/vehicle.png) |
| vowel | 990 | 13 | 0 | 13 | 11 | 0 | ![vowel](figures/vowel.png) |
| wdbc | 569 | 30 | 0 | 30 | 2 | 0 | ![wdbc](figures/wdbc.png) |
| wine | 178 | 13 | 0 | 13 | 3 | 0 | ![wine](figures/wine.png) |
| winequality-red | 1599 | 11 | 0 | 11 | 11 | 0 | ![winequality-red](figures/winequality-red.png) |
| winequality-white | 4898 | 11 | 0 | 11 | 11 | 0 | ![winequality-white](figures/winequality-white.png) |
| wisconsin | 699 | 9 | 0 | 9 | 2 | 2.29% | ![wisconsin](figures/wisconsin.png) |
| yeast | 1484 | 8 | 0 | 8 | 10 | 0 | ![yeast](figures/yeast.png) |
| zoo | 101 | 16 | 16 | 0 | 7 | 0 | ![zoo](figures/zoo.png) |

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  