import numpy as np
import sklearn
import random 
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from numpy.random import permutation
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.neural_network import MLPClassifier


samplers_all = [
    # Oversampling methods:
    RandomOverSampler(random_state=seed), 
    SMOTE(random_state=seed),             
    ADASYN(random_state=seed),            
    BorderlineSMOTE(random_state=seed),
    SVMSMOTE(random_state=seed),
    
    # Undersampling methods:
    RandomUnderSampler(random_state=seed),
    ClusterCentroids(random_state=seed),
    NearMiss(version=1, random_state=seed),
    NearMiss(version=2, random_state=seed),
    NearMiss(version=3, random_state=seed),
    TomekLinks(random_state=seed),
    EditedNearestNeighbours(random_state=seed),
    RepeatedEditedNearestNeighbours(random_state=seed),
    AllKNN(random_state=seed),
    CondensedNearestNeighbour(random_state=seed),
    OneSidedSelection(random_state=seed),
    NeighbourhoodCleaningRule(random_state=seed),
    InstanceHardnessThreshold(random_state=seed),
    
    
    # Combos:
    SMOTEENN(random_state=seed),
    SMOTETomek(random_state=seed)

]
samplers_array_all = np.array(samplers_all)

#### Every dataset has been labelled using this sequence 
#### (flip_fraction,num_informative,class_separation,num_clusters,random_seed,num_features,num_classes,
####  num_repeated,num_redundant)

num_datapoints = 1000

flip_fraction = [0,0.01]
num_informative = [3,4,5]
class_separation = np.arange(0.35, 2.05, 0.2).tolist()
num_clusters = [1,2,3]

random_seed = 0 
num_features = 5
num_classes = 2
num_repeated = 0
num_redundant = 0



data = []
# data2 = []






X_train_datasets = []
# y_train_datasets = []
# X_test_datasets = []
# y_test_datasets = []


# c = 0
for f in flip_fraction:
    for num_i in num_informative:
        for cs in class_separation:
            for num_c in num_clusters:
                
                
                c = c+1
                X,y = make_classification(n_samples=num_datapoints, n_features=num_features, n_informative=num_i, 
                                    n_redundant=num_redundant, n_repeated=num_repeated, n_classes=num_classes, n_clusters_per_class=num_c,
                                       class_sep=cs,
                                   flip_y=f,weights=[0.9,0.1], random_state = random_seed)



                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                sss.get_n_splits(X, y)
                for train_index, test_index in sss.split(X, y):
                    Xtrain, Xtest = X[train_index], X[test_index]
                    ytrain, ytest = y[train_index], y[test_index]

                     
                    
                    X_train_datasets.append(Xtrain)
#                     y_train_datasets.append(y_train)
#                     X_test_datasets.append(X_test)
#                     y_test_datasets.append(y_test)


                   


#                     data = []
#                     for j in range(10):
#                         Xtrain = X_train_datasets[j]
#                         ytrain = y_train_datasets[j]
#                         Xtest = X_test_datasets[j]
#                         ytest = y_test_datasets[j]


#                     data.append(tuple(["Dataset-" + str(c),"","","","","","","","","","","","","","","","","","","",""]))
#                     data2.append(tuple(["Dataset-" + str(c),"","","","","","","","","","","","","","","","","","","",""]))

              

                    row = ["$P$"]
#                     print(" &","$P$", end="")
                    for sampler in samplers_array_all:
                        t = ""
#                         precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, RandomForestClassifier(max_depth=2, random_state=0), Xtrain, Xtest, ytrain, ytest)
#                         print(precision)
                        try:
                            precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 18,max_iter = 300, random_state=1), Xtrain, Xtest, ytrain, ytest)
#                             print(" &", round(precision,3), end="")
                            t = str(round(precision,3))
                        except:
#                             print(" &", "N/A", end="")
                            t = "N/A"

                        row.append(t)

                        
#                     print(row)
                    data.append(tuple(row))

#                     print("\\\\")


                    rowdash = ["$R$"]
#                     print(" &","$R$", end="")
                    for sampler in samplers_array_all:
                        t = ""
                        try:
                            precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 18,max_iter = 200, random_state=1), Xtrain, Xtest, ytrain, ytest)
#                             print(" &", round(recall,3), end="")
                            t = str(round(recall,3))
                        except:
#                             print(" &", "N/A", end="")
                            t = "N/A"

                        rowdash.append(t)

                    data.append(tuple(rowdash))
            
                    

#                     print("\\\\")

                    

                    row1 = ["F_1"]
#                     print(" &","$F_1$", end="")
                    for sampler in samplers_array_all:
                        t = ""
                        try:
                            precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 18,max_iter = 200, random_state=1), Xtrain, Xtest, ytrain, ytest)
#                             print(" &", round(f1,3), end="")
                            t = str(round(f1,3))
                        except:
#                             print(" &", "N/A", end="")
                            t = "N/A"

                        row1.append(t)

                    data.append(tuple(row1))

#                     print("\\\\")





                    row2 = ["G_mean"]
#                     print(" &","$G_\\text{M}$", end="")
                    for sampler in samplers_array_all:
                        t = ""
                        try:
                            precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 18,max_iter = 200, random_state=1), Xtrain, Xtest,ytrain, ytest)
#                             print(" &", round(gmean,3), end="")
                            t = str(round(gmean,3))
                        except:
#                             print(" &", "N/A", end="")
                            t = "N/A"

                        row2.append(t)

                    data.append(tuple(row2))
#                     print("\\\\")




                    row3 = ["AUC"]
#                     print(" &","AUC", end="")
                    for sampler in samplers_array_all:
                        t = ""
                        try:
                            precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 18,max_iter = 200, random_state=1), Xtrain, Xtest,ytrain, ytest)
#                             print(" &", round(rocauc,3), end="")
                            t = str(round(rocauc,3))
                        except:
#                             print(" &", "N/A", end="")
                            t = "N/A"

                        row3.append(t)


                    data.append(tuple(row3))
#                     print("\\\\")

                    row4 = ["kappa"]
#                     print(" &","$\kappa$", end="")
                    for sampler in samplers_array_all:
                        t =""
                        try:
                            precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 18,max_iter = 200, random_state=1), Xtrain, Xtest,ytrain, ytest)
#                             print(" &", round(kappa,3), end="")
                            t = str(round(kappa,3))
                        except:
#                             print(" &", "N/A", end="")
                            t = "N/A"

                        row4.append(t)

                    data.append(tuple(row4))
            
            
            

#                     print("\\\\")



#                     if j != 10-1:
#                     print("\\midrule")
# print("\\bottomrule\n\\end{longtable}"+"\\end{center} \n" +
#       "\\end{document}")
np.savetxt('E:\Internships_19\Internship(Summer_19)\Imbalanced_class_classification\Class_Imabalanced_Learning_Code\CIL Code\RESULTS\Dataset_metrics_162_datasets_nn6.csv', data, delimiter=',', fmt=['%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Data,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')


