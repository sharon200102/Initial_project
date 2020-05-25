import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import Projects.Crohn_project.Constants as Constants
from Code import Data_loader as DL
import Code.Plot as Plot

sys.path.append(r"C:\sharon\second_degree\microbiome")
import infra_functions.preprocess_grid as preprocess_grid
#download and import the biom library
from biom import load_table
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
from sklearn import linear_model


# First load the the data including the biom type file
biom_table = load_table(Constants.biom_file_path)
mapping_table=DL.files_read_csv_fails_to_data_frame(Constants.mapping_file_path,1)

# Change the taxonomy structure to fit Yoel's preprocess grid function
taxonomy=DL.files_read_csv_fails_to_data_frame(Constants.taxonomy_page_url).drop('Confidence',axis=1)
taxonomy=taxonomy.rename({'Taxon':'taxonomy'},axis=1).set_index('Feature ID').transpose()
# then switch the biom table to dataframe format.
microbiome=biom_table.to_dataframe(True).transpose()
microbiome.index.name='ID'
#------------------------------------------------------------------------------------------------------------------------

# Add the taxonomy row to the biom dataframe (Yoel's documentation)
merged_microbiome_taxonomy = pd.concat([microbiome,taxonomy])
merged_microbiome_taxonomy.dropna(axis=1,inplace=True)
dict={'taxonomy_level':6,'taxnomy_group':'mean','epsilon':1,'normalization':'log','z_scoring':False,'norm_after_rel':False,'pca':15,'std_to_delete':0}
dec_data,merged_microbiome_taxonomy_b_pca,_ =preprocess_grid.preprocess_data(merged_microbiome_taxonomy,dict,mapping_table,'Crohn_data',False)[0:3]
#------------------------------------------------------------------------------------------------------------------------
# Plot Correlation
merged_microbiome_taxonomy_b_pca_with_mapping_features=merged_microbiome_taxonomy_b_pca.merge(mapping_table[['SampleID','CRP_n']],left_index=True,right_on='SampleID').set_index('SampleID')
target=merged_microbiome_taxonomy_b_pca_with_mapping_features['CRP_n']
Plot.draw_rhos_calculation_figure(target,merged_microbiome_taxonomy_b_pca_with_mapping_features.drop('CRP_n',axis=1),'Correlation between bacteria and target',6,save_folder='Graphs\Correlation\Bacteria')
Plot.draw_rhos_calculation_figure(mapping_table['CRP_n'],mapping_table[Constants.mapping_file_numeric_columns_relvant_to_correlation],'Correlation between numeric features and target',6,save_folder='Graphs\Correlation\Others')
#------------------------------------------------------------------------------------------------------------------------
#visualization
# Merge with the mapping table to gather more info on each sample

dec_data_with_mapping_features=dec_data.merge(mapping_table,left_index=True,right_on='SampleID').set_index('SampleID')

Plot.relationship_between_features(dec_data,'relationship_between_features',title="relationship_between_features colored by group2",color=dec_data_with_mapping_features['Group2'])
Plot.relationship_between_features(dec_data,'relationship_between_features',color=dec_data_with_mapping_features['smoking'],title="relationship_between_features colored by smoking")
Plot.relationship_between_features(dec_data,'relationship_between_features',color=dec_data_with_mapping_features['family_background_of_crohns'],title="relationship_between_features colored by family_background")
#------------------------------------------------------------------------------------------------------------------------
# Regression

dec_data_with_no_nan=dec_data[dec_data_with_mapping_features['CRP_n']!='NA']
crp_n_no_nan=dec_data_with_mapping_features['CRP_n'][dec_data_with_mapping_features['CRP_n']!='NA']
test_loss = []
train_loss = []
range=[ 5+n/20 for n in range(1,101)]
for C in range:
    clf = linear_model.Ridge(alpha=C)
    result = cross_validate(clf,dec_data_with_no_nan,crp_n_no_nan,cv=Constants.N_SPLITS,return_train_score=True)
    train_loss.append(result['train_score'].mean())
    test_loss.append((result['test_score'].mean()))
fig,ax=plt.subplots()

ax.plot(range, train_loss, marker='o', markersize=5, label='Train')
ax.plot(range, test_loss, marker='o', markersize=5, label='Test')
ax.set_xlabel('regularization')
ax.set_ylabel('R2')
ax.legend()
plt.show()


