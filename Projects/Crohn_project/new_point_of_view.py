import os
from biom import load_table
from Code import Data_loader as DL
from Projects.Crohn_project import Constants
script_dir = os.path.dirname(os.path.abspath(__file__))
# First load the the data including the biom type file
biom_table = load_table(os.path.join(script_dir, Constants.biom_file_path))
mapping_table = DL.files_read_csv_fails_to_data_frame(os.path.join(script_dir, Constants.mapping_file_path), 1)
taxonomy = DL.files_read_csv_fails_to_data_frame(Constants.taxonomy_page_url).drop('Confidence', axis=1)
taxonomy = taxonomy.rename({'Taxon': 'taxonomy'}, axis=1).set_index('Feature ID').transpose()
# then switch the biom table to dataframe format.
microbiome = biom_table.to_dataframe(True).transpose()
crohn_samples=mapping_table.loc[mapping_table['Group']!='Control']
crohn_patient_groups=crohn_samples.groupby(by='patient_No')
sum=0
for name,patient_samples in crohn_patient_groups:
    if patient_samples['Stool_number'].max()>'1':
        sum+=1
print(sum)