import matplotlib.pyplot as plt
import pandas as pd
import sys
import Projects.Crohn_project.Constants as Constants
from Code import Data_loader as DL
sys.path.append(r"C:\sharon\second_degree\microbiome")
import infra_functions.preprocess_grid as preprocess_grid
#download and import the biom library
from biom import load_table

# First load the the data including the biom type file
biom_table = load_table(Constants.biom_file_path)
mapping_table=DL.files_read_csv_fails_to_data_frame(Constants.mapping_file_path,1)
# Change the taxonomy structure to fit Yoel's preprocess grid function
taxonomy=DL.files_read_csv_fails_to_data_frame(Constants.taxonomy_page_url).drop('Confidence',axis=1)
taxonomy=taxonomy.rename({'Taxon':'taxonomy'},axis=1).set_index('Feature ID').transpose()


# then switch the biom table to dataframe format.
microbiome=biom_table.to_dataframe(True).transpose()
microbiome.index.name='ID'


# Add the taxonomy row to the biom dataframe (Yoel's documentation)
merged_microbiome_taxonomy = pd.concat([microbiome,taxonomy])
merged_microbiome_taxonomy.dropna(axis=1,inplace=True)
dict={'taxonomy_level':6,'taxnomy_group':'mean','epsilon':1,'normalization':'log','z_scoring':False,'norm_after_rel':False,'pca':5,'std_to_delete':0}
preprocess_grid.preprocess_data(merged_microbiome_taxonomy,dict,mapping_table,'Crohn_data',True)
