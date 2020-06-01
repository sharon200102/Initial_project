from pathlib import Path
taxonomy_page_url=r'https://raw.githubusercontent.com/sharon200102/Initial_project/master/Projects/Crohn_project/data/IIRN_taxonomy.tsv'
mapping_file_path= Path(r'data/IIRN_map_Yoram.txt')
biom_file_path=Path(r'data/IIRN_feature-table.biom')
N_SPLITS=2
mapping_file_numeric_columns_relvant_to_correlation=['months_since_start','Age','BMI','months_since_first_visit','Cal','CDAI','IBD_hospitalizations','flares_with_treatment_in_the_last_year','flares_with_treatment_since_diagnosis_per_year','HGB','weight_kg','disease_duration','Uri_MaRIAsc','Uri_Clermontsc','smoking_number_of_cigarettes_per_day', 'smoking_number_of_years','height_cm','date_of_diagnosis','age_at_diagnosis']
path_of_correlation_between_bacteria_and_target=Path('Graphs/Correlation/Bacteria')
path_of_correlation_between_numeric_features_and_target=Path('Graphs/Correlation/Others')