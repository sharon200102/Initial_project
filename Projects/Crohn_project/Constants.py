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
dec_data_path_to_save=Path('exports_for_learning_methods/dec_data.csv')
pca_obj_path_to_save=Path('exports_for_learning_methods/pca_obj.pkl')
mapping_file_with_tag_path_to_save=Path('exports_for_learning_methods/mapping_file_with_tag.csv')
tax = str(6)
k_fold = 5
test_size = 0.2
names = ["no anxiety", "anxiety"]
results_folder='results'
learning_method_parameters = {"TASK_TITLE": "Crohn_active_prediciton",  # the name of the task for plots titles...
            "FOLDER_TITLE": 'results',  # creates the folder for the task we want to do, save results in it
            "TAX_LEVEL": tax,
            "CLASSES_NAMES": names,
            "SVM": True,
            "SVM_params": {'kernel': ['linear'],
                           'gamma': ['auto', 'scale'],
                           'C': [0.01, 0.1, 1, 10, 100, 1000],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "sderot_anxiety"
                           },
            # if single option for each param -> single run, otherwise -> grid search.
            "XGB": True,
            "XGB_params": {'learning_rate': [0.1],
                           'objective': ['binary:logistic'],
                           'n_estimators': [1000],
                           'max_depth': [7],
                           'min_child_weight': [1],
                           'gamma': [1],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "sderot_anxiety"
                           },  # if single option for each param -> single run, otherwise -> grid search.
            "NN": True,
            "NN_params": {
                        "hid_dim_0": 120,
                        "hid_dim_1": 160,
                        "reg": 0.68,
                        "lr": 0.001,
                        "test_size": 0.1,
                        "batch_size": 32,
                        "shuffle": 1,
                        "num_workers": 4,
                        "epochs": 150,
                        "optimizer": 'SGD',
                        "loss": 'MSE',
                        "model": 'tanh_b'
            },  # if single option for each param -> single run, otherwise -> grid search.
            "NNI": False,
            "NNI_params": {
                        "result_type": 'auc'
            },
            # enter to model params?  might want to change for different models..
            "K_FOLD": k_fold,
            "TEST_SIZE": test_size,
            #  ...... add whatever
            }