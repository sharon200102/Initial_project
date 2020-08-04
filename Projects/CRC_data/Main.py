"""----Imports----"""
import pandas as pd
from Code import Data_loader as DL
import Projects.CRC_data.Constants as Constants
from Code.Preprocessing_actions import Structural_actions as SA
from Code.Decomposition import decompose
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import Code.Clustering as Clustering
import matplotlib.pyplot as plt
import Code.Plot as Plot
from sklearn.svm import SVR


"""Load the microbiome file"""
exported_features = DL.data_loader_exported_features(Constants.column_page_url, Constants.values_page_url)
# load and reduce the level of taxonomy file.
taxon = DL.files_read_csv_fails_to_data_frame(Constants.taxonomy_page_url).drop('Confidence',axis=1)
squashed_col = SA.decrease_level(taxon['Taxon'], Constants.LEVEL_TO_REDUCE, sep=';')
taxon = taxon.assign(Taxon=squashed_col)


# Merge the taxonomy with exported features and replace rows with same taxonomy with the mean of the rows.
merged_exported_features = pd.merge(exported_features, taxon, left_on='#OTU_ID',right_on='Feature ID').drop('Feature ID', axis=1)
reduced_exported_features = merged_exported_features.groupby('Taxon').mean()
"""In merged_table every sample has its corresponding microbiome"""
samples_results=SA.data_rearrangement(reduced_exported_features)
mapping_table=pd.read_csv(Constants.maping_page_url)
"""Because the deficiency in TP4 samples TP 5 will ve treated as TP4"""
mapping_table_with_new_time_target=mapping_table.assign(TimePointNum=mapping_table['TimePointNum'].replace(Constants.old_target_time,Constants.target_time))
merged_table=pd.merge(samples_results,mapping_table_with_new_time_target,on='#SampleID')
"""Get and select the wanted TimePoints"""
Group=input('Enter the groups that you would like to perform the analysis on  \n'+"\n".join(Constants.Groups_list)+"\n")
if Group =='CRC':
    merged_table=merged_table.loc[merged_table['Group']=='CRC']
    merged_table.reset_index(drop=True, inplace=True)
elif Group!='All':
    raise ValueError

"""
Add classification of T5 to all samples
identification is by CageNum MiceNum and Experiment 
"""
idx = 0
multi_class_tumor_load = []

for cage, mice, exp in zip(merged_table['CageNum'], merged_table['MiceNum'], merged_table['Experiment']):

    dic = {'CageNum': cage, 'MiceNum': mice, 'Experiment': exp, 'TimePointNum': Constants.target_time}
    item = SA.conditional_identification(merged_table, dic)['tumor_load'].values
    if (len(item) == 0):
        merged_table.drop(idx, inplace=True)
    else:
        multi_class_tumor_load.append(item[0])
    idx += 1

multi_class_tumor_load = pd.Series(multi_class_tumor_load)
if Group=='All':
    binary_tumor_load=multi_class_tumor_load.apply(lambda x: 1 if x>0 else 0)

elif Group=='CRC':
    median=multi_class_tumor_load.median()
    binary_tumor_load=multi_class_tumor_load.apply(lambda x: 1 if x>median else 0)
merged_table.reset_index(drop=True,inplace=True)

"""
Drop all columns that are not related to bacterial information.
"""

samples_bacterial_data=merged_table.drop(Constants.mapping_and_general_info_columns,axis=1)
samples_bacterial_data_and_identification=merged_table.drop(Constants.mapping_and_general_info_columns_no_identification,axis=1)
"""remove bacteria that only consist of zero """
SA.removeZeroCols(samples_bacterial_data)
SA.removeZeroCols(samples_bacterial_data_and_identification)

correlation_plots=False
if correlation_plots:
    """Plot a dynamic graph"""

    biit=Plot.bacteria_intraction_in_time(samples_bacterial_data_and_identification,Constants.identification_columns,'TimePointNum')
    v = [100] * len(biit.relevant_columns)
    biit.plot(node_size_list=v,G_name='Bacteria_interaction_network',folder='Bacteria_interaction_network')
    biit.export_edges_to_csv('Edges of Bacteria_interaction_network ')

    """Plot correlation between the bacteria and the target, correlation between immune_system_features and the target """


    Plot.draw_rhos_calculation_figure(multi_class_tumor_load,samples_bacterial_data.drop('#SampleID',axis=1),'Correlation between bacteria and target',6,save_folder='Graphs\Correlation\Bacteria')
    Plot.draw_rhos_calculation_figure(merged_table['tumor_load'],merged_table[Constants.immune_system_features],'Correlation between immune system parameters and the target',6,save_folder='Graphs\Correlation\Others')


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

title='Progress in time of column attribute mean {fn}'.format(fn=dimension_fn_name)
p1=Plot.progress_in_time_of_column_attribute_mean(dec_data,merged_table['TimePointNum'],binary_tumor_load)
new_plot=p1.plot()
new_plot.set_title(title)
new_plot.set_xlabel('Time')
new_plot.set_ylabel('Mean Value')
plt.show()
"""Look only at time point zero"""
dec_data_at0=dec_data[merged_table['TimePointNum']==0]

kind_of_prediction=input('Enter whether you want \n'+"\n".join(Constants.prediction_sort)+"\n")

if kind_of_prediction=='Binary':
    class_list_at0=binary_tumor_load[merged_table['TimePointNum']==0]
elif kind_of_prediction=='Multi':
    class_list_at0=multi_class_tumor_load[merged_table['TimePointNum']==0]
else:
    raise ValueError
"""Split the data and classify it"""
kf = KFold(n_splits = Constants.N_SPLITS, shuffle = True,random_state=5)
"""Get an evaluation function from the user"""
eval_fn_name=input('Enter the evaluation function wanted \n'+'\n'.join(Constants.Evaluation_name_list)+'\n')

percentage=[]
"""Evaluate the KNN for various K values. """
for neighbors in range(1,Constants.MAX_NIGH):
    neigh = KNeighborsClassifier(n_neighbors=neighbors,weights='distance',p=1)
    percentage.append(Clustering.cross_validation(kf,dec_data_at0,class_list_at0,neigh,eval_fn_name)[1])


"""Plot the results"""
plt.plot(range(1,Constants.MAX_NIGH),percentage,marker='o',markersize=15,markerfacecolor='red')
plt.title('Model : {pred} KNN \nDimension_fn : {dim} \nCorrelation_threshold : {'
'corr_TH}\nNormalization_fn : {norm}\n'.format(pred=kind_of_prediction,
                                                              dim=dimension_fn_name,corr_TH=str(Constants.THRESHOLD),
                                                              norm=normalization_fn_name))
plt.ylabel(eval_fn_name)
plt.xlabel('K neighbors')
plt.tight_layout()
plt.show()

