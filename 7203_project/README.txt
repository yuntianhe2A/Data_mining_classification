# Random Forest is used to perform classification, 1350 total instances used for training, get 0.89 cross_validation f1_score
1. You may Check the s4722166.csv results and delete it before running main.py
2. Run main.py directly to
    1. Pre-process Ecoli.csv(imputation/outliers removing/feature selection)
    2. 10-folds cross validation of random forest and print results
    3. Train on all 1350 instances and get a classifier
    4. Read Ecoli_test.csv and predict labels
    5. Write labels into s4722166.csv

# Environment:
	Python 3.10
	Pip: 22.2.2


# Libraries:
	pandas 1.4.4
	numpy 1.23.3
	scikit-learn 1.1.2

# Selected features for training: ['Nom (Col 106)', 'Num (Col 4)', 'Num (Col 95)', 'Num (Col 27)', 'Num (Col 35)', 'Num (Col 46)','Num (Col 66)', 'Num (Col 102)']

#Methods introduction:
    #pre_processing():perform class specific imputation and LOF_outliers detection to get 1350 examples for training

    #predict_test(): run random_forest() two times with different evaluation metrics accuracy/f1, train the classifier with whole training set and
			predict the labels for the test set with this classifier.

    #random_forest(): return cross validation results with random_forest classifier(300 trees, max_leaf_nodes=2)

    #LOF_outliers(): Detect and remove 16 outliers for class 1 and 134 outliers for class 0 from training set, it can improve
f1 score of the classifier from 0.82 to 0.87+, 
    #Removed outliers:
[2, 8, 12, 53, 55, 57, 58, 65, 70, 88, 108, 114, 121, 135, 137, 138, 141, 144, 146, 171, 176, 205, 213, 233, 243, 255, 257, 316, 323, 325, 335, 344, 360, 381, 416, 426, 427, 435, 437, 441, 446, 447, 451, 452, 467, 479, 483, 485, 503, 509, 523, 527, 529, 531, 535, 543, 544, 561, 585, 603, 607, 614, 617, 629, 633, 634, 639, 650, 670, 684, 688, 734, 743, 750, 768, 772, 827, 848, 867, 875, 876, 900, 912, 946, 962, 970, 971, 973, 982, 985, 989, 1001, 1002, 1010, 1033, 1042, 1043, 1045, 1053, 1057, 1087, 1090, 1094, 1108, 1111, 1113, 1115, 1146, 1152, 1163, 1175, 1180, 1185, 1187, 1194, 1215, 1224, 1235, 1315, 1350, 1361, 1375, 1379, 1383, 1386, 1393, 1399, 1400, 1404, 1420, 1458, 1476, 1478, 1498, 97, 362, 511, 554, 582, 601, 656, 657, 736, 825, 1245, 1260, 1264, 1291, 1327, 1454]
Total examples after removing outliers:  1350

    #impute(): perform imputation for columns 1-103(numerical) and 104-106 (norminal) seperately.

    #write_output(): write test set predictions and cross validation results into output file.
