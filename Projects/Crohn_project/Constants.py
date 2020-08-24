from pathlib import Path
taxonomy_page_url=r'https://raw.githubusercontent.com/sharon200102/Initial_project/master/Projects/Crohn_project/data/IIRN_taxonomy.tsv'
mapping_file_path= Path(r'data/IIRN_map_Yoram.txt')
biom_file_path=Path(r'data/IIRN_feature-table.biom')
N_SPLITS=2
mapping_file_numeric_columns_relvant_to_correlation=['months_since_start','Age','BMI','months_since_first_visit','Cal','CDAI','IBD_hospitalizations','flares_with_treatment_in_the_last_year','flares_with_treatment_since_diagnosis_per_year','HGB','weight_kg','disease_duration','Uri_MaRIAsc','Uri_Clermontsc','smoking_number_of_cigarettes_per_day', 'smoking_number_of_years','height_cm','date_of_diagnosis','age_at_diagnosis']
path_of_correlation_between_bacteria_and_target=Path('Graphs/Correlation/Bacteria')
path_of_correlation_between_numeric_features_and_target=Path('Graphs/Correlation/Others')
numerical_features_names=['CRP_n','Cal_n','CDAI_n','highest_lewis_score','HGB']
categorical_features_names=['family_background_of_crohns','smoking_binary','Group']
active_dict={'active':1}
control_dict={'Control':1}

dec_data_path_to_save=Path('exports_for_learning_methods/dec_data.csv')
pca_obj_path_to_save=Path('exports_for_learning_methods/pca_obj.pkl')
mapping_file_with_tag_path_to_save=Path('exports_for_learning_methods/mapping_file_with_tag.csv')
hidden_size=10
nn_structure=[15,hidden_size]
output_layer_size=2
lr=0.01
train_batch_size=32
val_batch_size=32
test_batch_size=32
epochs=25
active_classes_names=['Not Active','Active']
MODEL_PATH=Path(r'Model/flared_learning_model.pt')