import os
from biom import load_table
from Code import Data_loader as DL
from Projects.Crohn_project import Constants
from Code.Preprocessing_actions import Structural_actions as SA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
script_dir = os.path.dirname(os.path.abspath(__file__))

# load biom table
biom_table = load_table(os.path.join(script_dir, Constants.biom_file_path))
microbiome = biom_table.to_dataframe(True)

#load mapping file
mapping_table = DL.files_read_csv_fails_to_data_frame(os.path.join(script_dir, Constants.mapping_file_path), 1)

#load and squash the taxonomy file
taxonomy = DL.files_read_csv_fails_to_data_frame(Constants.taxonomy_page_url).drop('Confidence', axis=1)
taxonomy = taxonomy.rename({'Taxon': 'taxonomy'}, axis=1).set_index('Feature ID')
squashed_col = SA.decrease_level(taxonomy['taxonomy'], level=6, sep=';')
taxonomy = taxonomy.assign(taxonomy=squashed_col)


# then merge between the microbiome data-set and taxonomy to reduce dimensionality.
merged_microbiome_and_taxonomy = pd.merge(microbiome, taxonomy, left_index=True,right_index=True)
reduced_microbiome_and_taxonomy = merged_microbiome_and_taxonomy.groupby('taxonomy').mean()
bacteria_of_samples=SA.data_rearrangement(reduced_microbiome_and_taxonomy)
bacteria_of_samples.set_index('#SampleID',inplace=True)


crohn_samples=mapping_table.loc[mapping_table['Group']!='Control']
crohn_samples_of_patience_with_more_than_one_sample = crohn_samples[crohn_samples.duplicated('patient_No', keep=False)]
crohn_samples_of_patience_with_more_than_one_sample.set_index('SampleID',inplace=True)

merged_table=pd.merge(bacteria_of_samples,crohn_samples_of_patience_with_more_than_one_sample[['patient_No','Stool_number']],left_index=True,right_index=True)
merged_table.sort_values('Stool_number',inplace=True)
merged_table.drop('Stool_number',axis=1,inplace=True)
samples_grouped_by_patient = merged_table.groupby('patient_No',group_keys=False)
fig, patients_ax = plt.subplots(ncols=20)
fig.subplots_adjust(wspace=0.1)

for (patient_name,patient_samples),patient_ax in zip(samples_grouped_by_patient, patients_ax):
    sns.heatmap(patient_samples.loc[:, patient_samples.columns != 'patient_No'].transpose(), cmap="magma", ax=patient_ax, cbar=False)
    patient_ax.set_yticks([])
fig.suptitle('Patients microbiome development over time')
plt.show()

