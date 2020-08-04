import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Projects.Crohn_project.Constants as Constants
from Code import Data_loader as DL
import Code.Plot as Plot
import Preprocess.preprocess_grid as preprocess_grid
# download and import the biom library
from biom import load_table
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
from sklearn import linear_model
import os
from LearningMethods.general_functions import train_test_split
import seaborn as sns
import pickle
from LearningMethods.multi_model_learning import MultiModel

# In this script we will use paths that are relative to the main script absolute path.
script_dir = os.path.dirname(os.path.abspath(__file__))

# First load the the data including the biom type file
biom_table = load_table(os.path.join(script_dir, Constants.biom_file_path))
mapping_table = DL.files_read_csv_fails_to_data_frame(os.path.join(script_dir, Constants.mapping_file_path), 1)
# ------------------------------------------------------------------------------------------------------------------------
# visualize the rea;tionship between the categorical features and the continuous features in one plot.
visualize = False
if visualize == True:
    vis1 = Plot.categorical_vs_numeric_features_plot(mapping_table, Constants.categorical_features_names,
                                                     Constants.numerical_features_names, figsize=(15, 15))
    fig, grid = vis1.map_grid(hue_dict={('Group', 'CRP_n'): 'family_background_of_crohns'})
    for row in grid:
        for ax in row:
            ax.tick_params(axis="x", labelsize=6.5)
    plt.show()
# Change the taxonomy structure to fit Yoel's preprocess grid function
taxonomy = DL.files_read_csv_fails_to_data_frame(Constants.taxonomy_page_url).drop('Confidence', axis=1)
taxonomy = taxonomy.rename({'Taxon': 'taxonomy'}, axis=1).set_index('Feature ID').transpose()
# then switch the biom table to dataframe format.
microbiome = biom_table.to_dataframe(True).transpose()
microbiome.index.name = 'ID'
# ------------------------------------------------------------------------------------------------------------------------

# Add the taxonomy row to the biom dataframe (Yoel's documentation)
merged_microbiome_taxonomy = pd.concat([microbiome, taxonomy])
merged_microbiome_taxonomy.dropna(axis=1, inplace=True)
dict = {'taxonomy_level': 6, 'taxnomy_group': 'mean', 'epsilon': 1, 'normalization': 'log', 'z_scoring': False,
        'norm_after_rel': False, 'pca': (15, "pca"), 'std_to_delete': 0}
dec_data, merged_microbiome_taxonomy_b_pca, pca_obj, bacteria, _number_of_components = preprocess_grid.preprocess_data(
    merged_microbiome_taxonomy, dict, mapping_table, False)
bacteria = list(bacteria)
# ------------------------------------------------------------------------------------------------------------------------
# Get from the user the form of the analysis
target_feature = input('\n What feature you would like to use as a target\n')
# ------------------------------------------------------------------------------------------------------------------------
if target_feature == 'Group2':
    mapping_table.rename({target_feature: 'Tag'}, axis=1, inplace=True)
    mapping_table = mapping_table.assign(
        Tag=mapping_table['Tag'].transform(lambda status: Constants.active_dict.get(status, 0)))
    dec_data, target_df = preprocess_grid.adjust_table_to_target(dec_data, mapping_table, right_on='SampleID',
                                                                 left_index=True, remove_nan=True)
    train_idx_list, test_idx_list = train_test_split(dec_data, target_df['Tag'], target_df['patient_No'],
                                                     random_state=1)
    learning_method_parameters = Constants.learning_method_parameters
    learning_method_parameters['Bacteria'] = bacteria
    learning_method_parameters['PCA'] = pca_obj
    mm = MultiModel(learning_method_parameters)
    mm.fit(dec_data, target_df['Tag'], train_idx_list, test_idx_list, train_idx_list, test_idx_list)
else:
    raise Exception('The target feature {target} is not supported'.format(target=target_feature))
