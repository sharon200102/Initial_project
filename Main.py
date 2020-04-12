"""----Imports----"""
import pandas as pd
import Data_loader as DL
import Constants
from Preprocessing_actions import Structural_actions as SA
from Preprocessing_actions import Normalization
from Decomposition import decompose
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import Clustering
import matplotlib.pyplot as plt
from sklearn import svm
import Plot

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
"""
Add classification of T5 to all samples
identification is by CageNum MiceNum and Experiment 
"""

idx = 0
class_list = []

for cage, mice, exp in zip(merged_table['CageNum'], merged_table['MiceNum'], merged_table['Experiment']):

    dic = {'CageNum': cage, 'MiceNum': mice, 'Experiment': exp, 'TimePointNum': 5}
    item = SA.conditional_identification(mapping_table, dic)['tumor_load'].values
    if (len(item) == 0):
        merged_table.drop(idx, inplace=True)
    else:
        class_list.append(1) if item > 0 else class_list.append(0)
    idx += 1
class_list = pd.Series(class_list)

"""
Before drooping columns, Save categorical columns that will be important in the future.
"""
merged_table.reset_index(drop=True,inplace=True)
relevant_categorical=merged_table[Constants.relevant_categorical_names]
relevant_categorical=relevant_categorical.assign(tumor_load=class_list.values)
"""
Drop unnecessary columns.
"""
merged_table.drop(Constants.columns_to_be_dropped,inplace=True,axis=1)
SA.removeZeroCols(merged_table)

"""Remove highly correlated columns"""
uncorr_data=SA.dropHighCorr(merged_table,Constants.THRESHOLD)

"""Get an Normalization function from the user and normaliaze the data according to it"""
normalization_fn_name=input('Enter the normalization function wanted \n Robust \n Zscore \n MinMax\n')
normalized_data= Constants.normalization_dict[normalization_fn_name](uncorr_data)

"""Dimension reduction on the whole data"""
dimension_fn_name=input('Enter the dimensionality reduction function wanted \n PCA \n ICA \n')
dec_obj,dec_data=decompose(normalized_data,Constants.dimension_reduction_dict[dimension_fn_name],n_components=5,random_state=1)
"""Visualizations after decomposition"""
Plot.visualize_in_pairs(dec_data,"decomposed_data",relevant_categorical)
Plot.column_attribute_progress_in_categorical(dec_data,relevant_categorical['TimePointNum'],"Time",splitter=relevant_categorical['tumor_load'],splitter_name="tumor_load")
"""Look only at time point zero"""
dec_data_at0=dec_data[relevant_categorical['TimePointNum']==0]
class_list_at0=class_list[relevant_categorical['TimePointNum']==0]


"""Split the data and cluster it"""
kf = KFold(n_splits = Constants.N_SPLITS, shuffle = True,random_state=5)

percentage=[]
score_per_fold=[]
for neighbors in range(1,Constants.MAX_NIGH):
    neigh = KNeighborsClassifier(n_neighbors=neighbors,weights='distance',p=1)
    percentage.append(Clustering.cross_validation(kf,dec_data_at0,class_list_at0,neigh))


"""Plot the results"""
plt.plot(range(1,Constants.MAX_NIGH),percentage,marker='o',markersize=15,markerfacecolor='red')
plt.title('Model : KNN \n'+'Dimension_fn : '+str(dimension_fn_name)+' on all time points'+'\n'+'Correlation_threshold : '+str(Constants.THRESHOLD)+'\n Normalization_fn : '+str(normalization_fn_name+'\n'))
plt.xlabel('K neighbors')
plt.ylabel('Accuracy ratio')
plt.tight_layout()
plt.show()
