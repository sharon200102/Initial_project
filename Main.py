"""----Imports----"""
import pandas as pd
import Data_loader as DL
import Constants
from Preprocessing_actions import Structural_actions as SA
from Decomposition import decompose
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import Clustering
import matplotlib.pyplot as plt
import Plot
from sklearn.svm import SVR
import seaborn as sns
from sklearn import linear_model
"""Load the microbiome file"""
exported_features = DL.data_loader_exported_features(Constants.column_page_url, Constants.values_page_url)
# load and reduce the level of taxonomy file.
taxon = DL.taxonomy(Constants.taxonomy_page_url)
squashed_col = SA.decrease_level(taxon['Taxon'], Constants.LEVEL_TO_REDUCE, sep=';')
taxon = taxon.assign(Taxon=squashed_col)


# Merge the taxonomy with exported features and replace rows with same taxonomy with the mean of the rows.
merged_exported_features = pd.merge(exported_features, taxon, on='#OTU_ID')
reduced_exported_features = merged_exported_features.groupby('Taxon').mean()

"""In merged_table every sample has its corresponding microbiome"""
samples_results=SA.data_rearrangement(reduced_exported_features)
mapping_table=pd.read_csv(Constants.maping_page_url)
merged_table=pd.merge(samples_results,mapping_table,on='#SampleID')

"""Get and select the wanted TimePoints"""
TimePoint=input('Enter the time points to perform on  \n'+"\n".join(Constants.TimePoints_list)+"\n")
if TimePoint=='0':
    merged_table=merged_table[merged_table['TimePointNum']==0]
elif TimePoint!='All':
    raise ValueError

"""
Add classification of T5 to all samples
identification is by CageNum MiceNum and Experiment 
"""

idx = 0
multi_class_tumor_load = []

for cage, mice, exp in zip(merged_table['CageNum'], merged_table['MiceNum'], merged_table['Experiment']):

    dic = {'CageNum': cage, 'MiceNum': mice, 'Experiment': exp, 'TimePointNum': 5}
    item = SA.conditional_identification(mapping_table, dic)['tumor_load'].values
    if (len(item) == 0):
        merged_table.drop(idx, inplace=True)
    else:
        multi_class_tumor_load.append(item[0])
    idx += 1

multi_class_tumor_load = pd.Series(multi_class_tumor_load)
binary_tumor_load=multi_class_tumor_load.apply(lambda x: 1 if x>0 else 0)

merged_table.reset_index(drop=True,inplace=True)
"""
Drop all columns that are not related to bacterial information.
"""

samples_bacterial_data=merged_table.drop(Constants.mapping_and_general_info_columns,axis=1)
"""remove bacteria that only consist of zero """
SA.removeZeroCols(samples_bacterial_data)
"""Plot correlation between the bacteria and the target, correlation between immune_system_features and the target """

"""
Plot.draw_rhos_calculation_figure(multi_class_tumor_load,samples_bacterial_data.drop('#SampleID',axis=1),'Correlation between bacteria and target',6,save_folder='Graphs\Correlation\Bacteria')
Plot.draw_rhos_calculation_figure(merged_table['tumor_load'],merged_table[Constants.immune_system_features],'Correlation between immune system parameters and the target',6,save_folder='Graphs\Correlation\Others')
"""

"""Remove highly correlated columns"""
uncorr_data=SA.dropHighCorr(samples_bacterial_data,Constants.THRESHOLD)

"""Get an Normalization function from the user and normaliaze the data according to it"""
normalization_fn_name=input('Enter the normalization function wanted \n'+"\n".join(Constants.normalization_dict.keys())+"\n")
normalized_data= Constants.normalization_dict[normalization_fn_name](uncorr_data, *Constants.normalization_parameters_dict[normalization_fn_name])

"""Dimension reduction"""
dimension_fn_name=input('Enter the dimensionality reduction function wanted \n PCA \n ICA \n')
dec_obj,dec_data=decompose(normalized_data,Constants.dimension_reduction_dict[dimension_fn_name],n_components=5,random_state=1)

"""Visualizations after decomposition, different visualizations will be performed based on initially selected time points"""
labels_dict={1:'Tumor',0:'No tumor'}
title='Data after {fn} relationship between features '.format(fn=dimension_fn_name)
Plot.relationship_between_features(dec_data,folder='Graphs',color=binary_tumor_load,title=title,labels_dict=labels_dict)

if TimePoint=='All':
    title='Progress in time of column attribute mean {fn}'.format(fn=dimension_fn_name)
    p1=Plot.progress_in_time_of_column_attribute_mean(dec_data,merged_table['TimePointNum'],binary_tumor_load)
    new_plot=p1.plot()
    new_plot.set_title(title)
    new_plot.set_xlabel('Time')
    new_plot.set_ylabel('Mean')
    plt.show()
"""Look only at time point zero"""
dec_data_at0=dec_data[merged_table['TimePointNum']==0]

kind_of_prediction=input('Enter whether you want \n'+"\n".join(Constants.prediction_sort)+"\n")

if kind_of_prediction=='Binary':
    class_list_at0=binary_tumor_load[merged_table['TimePointNum']==0]
elif kind_of_prediction=='Multi' or kind_of_prediction=='Regression' :
    class_list_at0=multi_class_tumor_load[merged_table['TimePointNum']==0]
else:
    raise ValueError
"""Split the data and classify it"""
kf = KFold(n_splits = Constants.N_SPLITS, shuffle = True,random_state=5)
"""Get an evaluation function from the user"""
eval_fn_name=input('Enter the evaluation function wanted \n'+'\n'.join(Constants.Evaluation_name_list)+'\n')

if kind_of_prediction!='Regression':
    percentage=[]
    """Evaluate the KNN for various K values. """
    for neighbors in range(1,Constants.MAX_NIGH):
        neigh = KNeighborsClassifier(n_neighbors=neighbors,weights='distance',p=1)
        percentage.append(Clustering.cross_validation(kf,dec_data_at0,class_list_at0,neigh,eval_fn_name)[1])


    """Plot the results"""
    plt.plot(range(1,Constants.MAX_NIGH),percentage,marker='o',markersize=15,markerfacecolor='red')
    plt.title('Data TimePoints : {TP}\nModel : {pred} KNN \nDimension_fn : {dim} \nCorrelation_threshold : {'
'corr_TH}\nNormalization_fn : {norm}\n'.format(TP=TimePoint,pred=kind_of_prediction,
                                                              dim=dimension_fn_name,corr_TH=str(Constants.THRESHOLD),
                                                              norm=normalization_fn_name))
    plt.ylabel(eval_fn_name)
    plt.xlabel('K neighbors')
    plt.tight_layout()
    plt.show()

    """if the user wish to perform a regression, build a SVR model and print the R2 performance"""

elif kind_of_prediction=='Regression':
    test_loss=[]
    train_loss=[]
    for C in [0+p/100 for p in range(1,10000)]:
        clf= SVR(C=C, epsilon=0.1,kernel='poly',degree=3,gamma='scale')
        result=Clustering.cross_validation(kf,dec_data_at0,class_list_at0,clf,eval_fn_name)
        train_loss.append(result[0])
        test_loss.append((result[1]))
    plt.plot([0+p/100 for p in range(1,10000)],train_loss,marker='o',markersize=5,label='Train')
    plt.plot([0+p/100 for p in range(1,10000)],test_loss,marker='o',markersize=5,label='Test')
    plt.xlabel('regularization')
    plt.ylabel('R2')
    plt.legend()
    plt.show()