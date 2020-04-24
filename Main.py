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

exported_features = DL.data_loader_exported_features(Constants.column_page_url, Constants.values_page_url)
# load and squash taxonomy file.
taxon = DL.taxonomy(Constants.taxonomy_page_url)
squashed_col = SA.decrease_level(taxon['Taxon'], 6, sep=';')
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

"""
Before drooping columns, Save categorical columns that will be important in the future.
"""
merged_table.reset_index(drop=True,inplace=True)

relevant_categorical=merged_table[Constants.relevant_categorical_names]
relevant_categorical=relevant_categorical.assign(binary_tumor_load=binary_tumor_load.values)
"""
Drop unnecessary columns.
"""
merged_table.drop(Constants.columns_to_be_dropped,inplace=True,axis=1)
SA.removeZeroCols(merged_table)

"""Remove highly correlated columns"""
uncorr_data=SA.dropHighCorr(merged_table,Constants.THRESHOLD)

"""Get an Normalization function from the user and normaliaze the data according to it"""
normalization_fn_name=input('Enter the normalization function wanted \n'+"\n".join(Constants.normalization_dict.keys())+"\n")
normalized_data= Constants.normalization_dict[normalization_fn_name](uncorr_data, *Constants.normalization_parameters_dict[normalization_fn_name])

"""Dimension reduction"""
dimension_fn_name=input('Enter the dimensionality reduction function wanted \n PCA \n ICA \n')
dec_obj,dec_data=decompose(normalized_data,Constants.dimension_reduction_dict[dimension_fn_name],n_components=5,random_state=1)
"""Visualizations after decomposition, different visualizations will be performed based on initially selected time points"""
Plot.relationship_between_features(dec_data,folder='Graphs',color=relevant_categorical['binary_tumor_load'])

if TimePoint=='All':
    Plot.progress_in_time_of_column_attribute_mean(dec_data,relevant_categorical['TimePointNum'],"Time",attribute_series=relevant_categorical['binary_tumor_load'],splitter_name="binary_tumor_load")
    Plot.t_test_progress_over_categorical(dec_data,relevant_categorical['TimePointNum'],relevant_categorical['binary_tumor_load'],"Time","binary_tumor_load")

"""Look only at time point zero"""
dec_data_at0=dec_data[relevant_categorical['TimePointNum']==0]

kind_of_prediction=input('Enter whether you want \n'+"\n".join(Constants.prediction_sort)+"\n")

if kind_of_prediction=='Binary':
    class_list_at0=binary_tumor_load[relevant_categorical['TimePointNum']==0]
else:
    class_list_at0=multi_class_tumor_load[relevant_categorical['TimePointNum']==0]

"""Split the data and classify it"""
kf = KFold(n_splits = Constants.N_SPLITS, shuffle = True,random_state=5)
"""Get an evaluation function from the user"""
eval_fn_name=input('Enter the evaluation function wanted \n'+'\n'.join(Constants.Evaluation_name_list)+'\n')

if kind_of_prediction!='Regression':
    percentage=[]
    for neighbors in range(1,Constants.MAX_NIGH):
        neigh = KNeighborsClassifier(n_neighbors=neighbors,weights='distance',p=1)
        percentage.append(Clustering.cross_validation(kf,dec_data_at0,class_list_at0,neigh,eval_fn_name))


    """Plot the results"""
    plt.plot(range(1,Constants.MAX_NIGH),percentage,marker='o',markersize=15,markerfacecolor='red')
    plt.title('Data TimePoints : '+TimePoint+'\n'+'Model : '+kind_of_prediction+' KNN \n'''+'Dimension_fn : '+str(dimension_fn_name)+'\n'+'Correlation_threshold : '+str(Constants.THRESHOLD)+'\n Normalization_fn : '+str(normalization_fn_name+'\n'))
    plt.ylabel(eval_fn_name)
    plt.xlabel('K neighbors')
    plt.tight_layout()
    plt.show()

    """if the user wish to perform a regression, build a SVR model and print the R2 performance"""

elif kind_of_prediction=='Regression':
    clf = SVR(C =4,epsilon=0.0001,gamma='scale')
    print(Clustering.cross_validation(kf,dec_data_at0,class_list_at0,clf,eval_fn_name))