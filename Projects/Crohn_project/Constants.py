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
names = ["Not Active", "Active"]
results_folder='results'
learning_method_parameters = {"TASK_TITLE": "Crohn_active_prediciton",  # the name of the task for plots titles...
            "FOLDER_TITLE": 'results',  # creates the folder for the task we want to do, save results in it
            "TAX_LEVEL": tax,
            "CLASSES_NAMES": names,
            "SVM": False,
            "SVM_params": {'kernel': ['linear','poly','sigmoid'],
                           'gamma': ['auto', 'scale'],
                           'C': [0.01*5**i for i in range(0,11)],
                           "create_coeff_plots": False,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "Active_prediction"
                           },
            # if single option for each param -> single run, otherwise -> grid search.
            "XGB": False,
            "XGB_params": {'learning_rate': [0.3,0.4,0.5],
                           'objective': ['binary:logistic'],
                           'n_estimators': [500,1000],
                           'max_depth': [6,7,8],
                           'min_child_weight': [1,2,3,4],
                           'gamma': [0.001,0.01,0.1],
                           "create_coeff_plots": False,
                           "CLASSES_NAMES": names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "Active_prediction"
                           },  # if single option for each param -> single run, otherwise -> grid search.
            "NN": True,
            "NN_params": {
                        "hid_dim_0": 50,
                        "hid_dim_1": 50,
                        "reg": 0.32,
                        "lr": 0.01,
                        "test_size": 0.2,
                        "batch_size": 16,
                        "shuffle": 1,
                        "num_workers": 0,
                        "epochs": 20,
                        "optimizer": 'Adam',
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
            "Folder":"models_results"
            #  ...... add whatever
            }