## KEEL Datasets

Most of the datasets were extracted from KEEL repository. Some of them present rows with missing values (which is the data present at the column missing), All datasets are in .arff format and separated in 20 files, a train and test set for each one of the iterations of a 10-fold stratified cross validation.

|name | instances | attributes | categorical | numeric | classes | missing | class distribution |
| -----: |  -----: |  -----: |  -----: |  -----: |  -----: |  -----: |  -----: |
|abalone | 4174 | 9 | 1 | 8 | 28 | 0.0 | ![abalone](figures/abalone.png) |
|adult | 48842 | 15 | 6 | 9 | 2 | 0.0 | ![adult](figures/adult.png) |
|appendicitis | 106 | 8 | 1 | 7 | 2 | 0.0 | ![appendicitis](figures/appendicitis.png) |
|artificialcharacters | 10218 | 8 | 1 | 7 | 10 | 0.0 | ![artificialcharacters](figures/artificialcharacters.png) |
|australian | 690 | 15 | 6 | 9 | 2 | 0.0 | ![australian](figures/australian.png) |
|balance | 625 | 5 | 1 | 4 | 3 | 0.0 | ![balance](figures/balance.png) |
|balancescale | 625 | 5 | 1 | 4 | 3 | 0.0 | ![balancescale](figures/balancescale.png) |
|banana | 5300 | 3 | 1 | 2 | 2 | 0.0 | ![banana](figures/banana.png) |
|banknotes | 1372 | 5 | 1 | 4 | 2 | 0.0 | ![banknotes](figures/banknotes.png) |
|bloodtransfusion | 748 | 5 | 1 | 4 | 2 | 0.0 | ![bloodtransfusion](figures/bloodtransfusion.png) |
|breast | 286 | 10 | 5 | 5 | 2 | 0.0 | ![breast](figures/breast.png) |
|bupa | 345 | 7 | 1 | 6 | 2 | 0.0 | ![bupa](figures/bupa.png) |
|car | 1728 | 7 | 7 | 0 | 4 | 0.0 | ![car](figures/car.png) |
|cargofreight | 3943 | 93 | 2 | 91 | 4 | 1.0 | ![cargofreight](figures/cargofreight.png) |
|chess | 3196 | 37 | 36 | 1 | 2 | 0.0 | ![chess](figures/chess.png) |
|chorals | 5665 | 15 | 13 | 2 | 102 | 0.0 | ![chorals](figures/chorals.png) |
|cleveland | 303 | 14 | 1 | 13 | 5 | 0.02 | ![cleveland](figures/cleveland.png) |
|coil2000 | 9822 | 86 | 1 | 85 | 2 | 0.0 | ![coil2000](figures/coil2000.png) |
|collins | 1000 | 20 | 0 | 20 | 30 | 0.0 | ![collins](figures/collins.png) |
|connect-4 | 67557 | 43 | 43 | 0 | 3 | 0.0 | ![connect-4](figures/connect-4.png) |
|contraceptive | 1473 | 10 | 1 | 9 | 3 | 0.0 | ![contraceptive](figures/contraceptive.png) |
|creditapproval | 690 | 16 | 7 | 9 | 2 | 0.03 | ![creditapproval](figures/creditapproval.png) |
|crx | 690 | 16 | 6 | 10 | 2 | 0.03 | ![crx](figures/crx.png) |
|dermatology | 366 | 35 | 1 | 34 | 6 | 0.02 | ![dermatology](figures/dermatology.png) |
|diabetic | 1151 | 20 | 3 | 17 | 2 | 0.0 | ![diabetic](figures/diabetic.png) |
|drugconsumption | 1885 | 31 | 12 | 19 | 7 | 0.0 | ![drugconsumption](figures/drugconsumption.png) |
|dummygerman | 1800 | 21 | 14 | 7 | 2 | 0.0 | ![dummygerman](figures/dummygerman.png) |
|ecoli | 336 | 8 | 0 | 8 | 8 | 0.0 | ![ecoli](figures/ecoli.png) |
|electricity | 45312 | 9 | 2 | 7 | 2 | 0.0 | ![electricity](figures/electricity.png) |
|fars | 100968 | 30 | 15 | 15 | 8 | 0.0 | ![fars](figures/fars.png) |
|flare | 1066 | 12 | 8 | 4 | 6 | 0.0 | ![flare](figures/flare.png) |
|frogs | 7195 | 23 | 1 | 22 | 4 | 0.0 | ![frogs](figures/frogs.png) |
|german | 1000 | 21 | 14 | 7 | 2 | 0.0 | ![german](figures/german.png) |
|ginaagnostic | 3468 | 971 | 1 | 970 | 2 | 0.0 | ![ginaagnostic](figures/ginaagnostic.png) |
|glass | 214 | 10 | 1 | 9 | 6 | 0.0 | ![glass](figures/glass.png) |
|haberman | 306 | 4 | 1 | 3 | 2 | 0.0 | ![haberman](figures/haberman.png) |
|hayes-roth | 160 | 5 | 1 | 4 | 3 | 0.0 | ![hayes-roth](figures/hayes-roth.png) |
|hcvegypt | 1385 | 29 | 10 | 19 | 4 | 0.0 | ![hcvegypt](figures/hcvegypt.png) |
|heart | 270 | 14 | 1 | 13 | 2 | 0.0 | ![heart](figures/heart.png) |
|iBeacon | 1420 | 14 | 0 | 14 | 105 | 0.0 | ![iBeacon](figures/iBeacon.png) |
|ionosphere | 351 | 34 | 1 | 33 | 2 | 0.0 | ![ionosphere](figures/ionosphere.png) |
|iris | 150 | 5 | 1 | 4 | 3 | 0.0 | ![iris](figures/iris.png) |
|kddcup | 494020 | 42 | 8 | 34 | 23 | 0.0 | ![kddcup](figures/kddcup.png) |
|krvskp | 3196 | 37 | 36 | 1 | 2 | 0.0 | ![krvskp](figures/krvskp.png) |
|led7digit | 500 | 8 | 1 | 7 | 10 | 0.0 | ![led7digit](figures/led7digit.png) |
|letter | 20000 | 17 | 1 | 16 | 26 | 0.0 | ![letter](figures/letter.png) |
|lymphography | 148 | 19 | 10 | 9 | 4 | 0.0 | ![lymphography](figures/lymphography.png) |
|magic | 19020 | 11 | 1 | 10 | 2 | 0.0 | ![magic](figures/magic.png) |
|mammographic | 961 | 6 | 1 | 5 | 2 | 0.14 | ![mammographic](figures/mammographic.png) |
|monk-2 | 432 | 7 | 1 | 6 | 2 | 0.0 | ![monk-2](figures/monk-2.png) |
|movement_libras | 360 | 91 | 1 | 90 | 15 | 0.0 | ![movement_libras](figures/movement_libras.png) |
|newthyroid | 215 | 6 | 1 | 5 | 3 | 0.0 | ![newthyroid](figures/newthyroid.png) |
|optdigits | 5620 | 65 | 1 | 64 | 10 | 0.0 | ![optdigits](figures/optdigits.png) |
|page-blocks | 5472 | 11 | 1 | 10 | 5 | 0.0 | ![page-blocks](figures/page-blocks.png) |
|penbased | 10992 | 17 | 1 | 16 | 10 | 0.0 | ![penbased](figures/penbased.png) |
|phoneme | 5404 | 6 | 1 | 5 | 2 | 0.0 | ![phoneme](figures/phoneme.png) |
|pima | 768 | 9 | 1 | 8 | 2 | 0.0 | ![pima](figures/pima.png) |
|polishbanks | 10503 | 65 | 1 | 64 | 2 | 0.53 | ![polishbanks](figures/polishbanks.png) |
|post-operative | 90 | 9 | 3 | 6 | 3 | 0.0 | ![post-operative](figures/post-operative.png) |
|ring | 7400 | 21 | 1 | 20 | 2 | 0.0 | ![ring](figures/ring.png) |
|saheart | 462 | 10 | 2 | 8 | 2 | 0.0 | ![saheart](figures/saheart.png) |
|satimage | 6435 | 37 | 1 | 36 | 6 | 0.0 | ![satimage](figures/satimage.png) |
|segment | 2310 | 20 | 1 | 19 | 7 | 0.0 | ![segment](figures/segment.png) |
|seismicbumps | 2584 | 19 | 5 | 14 | 2 | 0.0 | ![seismicbumps](figures/seismicbumps.png) |
|semeion | 1593 | 257 | 0 | 257 | 11 | 0.0 | ![semeion](figures/semeion.png) |
|sonar | 208 | 61 | 1 | 60 | 2 | 0.0 | ![sonar](figures/sonar.png) |
|soybean | 683 | 36 | 34 | 2 | 19 | 0.0 | ![soybean](figures/soybean.png) |
|spambase | 4597 | 58 | 1 | 57 | 2 | 0.0 | ![spambase](figures/spambase.png) |
|spectfheart | 267 | 45 | 1 | 44 | 2 | 0.0 | ![spectfheart](figures/spectfheart.png) |
|splice | 3190 | 61 | 16 | 45 | 3 | 0.0 | ![splice](figures/splice.png) |
|steelfaults | 1941 | 34 | 10 | 24 | 2 | 0.0 | ![steelfaults](figures/steelfaults.png) |
|syntheticcontrol | 600 | 61 | 1 | 60 | 6 | 0.0 | ![syntheticcontrol](figures/syntheticcontrol.png) |
|tae | 151 | 6 | 1 | 5 | 3 | 0.0 | ![tae](figures/tae.png) |
|texture | 5500 | 41 | 1 | 40 | 11 | 0.0 | ![texture](figures/texture.png) |
|thyroid | 7200 | 22 | 1 | 21 | 3 | 0.0 | ![thyroid](figures/thyroid.png) |
|tic-tac-toe | 958 | 10 | 10 | 0 | 2 | 0.0 | ![tic-tac-toe](figures/tic-tac-toe.png) |
|tictactoe | 958 | 10 | 10 | 0 | 2 | 0.0 | ![tictactoe](figures/tictactoe.png) |
|titanic | 2201 | 4 | 1 | 3 | 2 | 0.0 | ![titanic](figures/titanic.png) |
|turkiye | 5820 | 33 | 33 | 0 | 13 | 0.0 | ![turkiye](figures/turkiye.png) |
|twonorm | 7400 | 21 | 1 | 20 | 2 | 0.0 | ![twonorm](figures/twonorm.png) |
|vehicle | 846 | 19 | 1 | 18 | 4 | 0.0 | ![vehicle](figures/vehicle.png) |
|vowel | 990 | 14 | 1 | 13 | 11 | 0.0 | ![vowel](figures/vowel.png) |
|waveform | 5000 | 41 | 1 | 40 | 3 | 0.0 | ![waveform](figures/waveform.png) |
|wdbc | 569 | 31 | 1 | 30 | 2 | 0.0 | ![wdbc](figures/wdbc.png) |
|wine | 178 | 14 | 1 | 13 | 3 | 0.0 | ![wine](figures/wine.png) |
|winequality-red | 1599 | 12 | 1 | 11 | 6 | 0.0 | ![winequality-red](figures/winequality-red.png) |
|winequality-white | 4898 | 12 | 0 | 12 | 7 | 0.0 | ![winequality-white](figures/winequality-white.png) |
|wisconsin | 699 | 10 | 1 | 9 | 2 | 0.02 | ![wisconsin](figures/wisconsin.png) |
|yeast | 1484 | 9 | 1 | 8 | 10 | 0.0 | ![yeast](figures/yeast.png) |
|zoo | 101 | 17 | 15 | 2 | 7 | 0.0 | ![zoo](figures/zoo.png) |
