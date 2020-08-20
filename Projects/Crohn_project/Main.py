import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix,precision_recall_curve,precision_recall_fscore_support
from Plot.plot_confusion_mat import print_confusion_matrix
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
import Code.Clustering as Clustering
from torch.utils.data import TensorDataset,DataLoader
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
    mapping_table_with_modified_target_name=mapping_table.rename({target_feature: 'Tag'}, axis=1)
    mapping_table_with_binary_target = mapping_table_with_modified_target_name.assign(Tag=mapping_table_with_modified_target_name['Tag'].transform(lambda status: Constants.active_dict.get(status, 0)))
    dec_data_adusted_to_target, target_df = preprocess_grid.adjust_table_to_target(dec_data, mapping_table_with_binary_target, right_on='SampleID',
                                                                 left_index=True, remove_nan=True)
    target_df.reset_index(drop=True,inplace=True)
    """Calculate the weights for the loss function"""
    target_unique_elements=sorted(list(target_df['Tag'].unique()))
    quantity_of_target_unique_elements=[list(target_df['Tag']).count(i) for i in target_unique_elements]
    normedWeights=[1/x for x in quantity_of_target_unique_elements]
    normedWeights = torch.FloatTensor(normedWeights)

    tensor_data = torch.from_numpy(dec_data_adusted_to_target.to_numpy()).type(torch.FloatTensor)
    tensor_target = torch.from_numpy(target_df['Tag'].to_numpy()).type(torch.LongTensor)

    """Create the model and all the surroundings"""
    flared_learning_model=Clustering.learning_model(Constants.nn_structure,Constants.output_layer_size)
    loss_fn=torch.nn.CrossEntropyLoss(weight=normedWeights,reduction='sum')
    optimizer=torch.optim.Adam(flared_learning_model.parameters(), lr=Constants.lr)
    train_step=Clustering.make_train_step(flared_learning_model,loss_fn,optimizer)




    train_idx_list, test_idx_list = train_test_split(dec_data_adusted_to_target, target_df['Tag'], target_df['patient_No'],return_itr=False,
                                                     random_state=1)



    for current_train_and_val_idx, current_test_idx in zip(train_idx_list, test_idx_list):
        """Split to train-validation and test"""
        x_train_and_val_tensor = tensor_data[current_train_and_val_idx]
        y_train_and_val_tensor = tensor_target[current_train_and_val_idx]

        """Split train-validation to train, validation"""
        train_idx,val_idx=train_test_split(x_train_and_val_tensor,y_train_and_val_tensor,target_df['patient_No'].iloc[current_train_and_val_idx],n_splits=1)
        x_train_tensor,y_train_tensor=tensor_data[train_idx],tensor_target[train_idx]
        x_val_tensor,y_val_tensor=tensor_data[val_idx],tensor_target[val_idx]

        """Test set"""
        x_test_tensor = tensor_data[current_test_idx]
        y_test_tensor = tensor_target[current_test_idx]

        train_and_val_data_set = TensorDataset(x_train_and_val_tensor, y_train_and_val_tensor)
        val_data_set=TensorDataset(x_val_tensor, y_val_tensor)
        test_data_set = TensorDataset(x_test_tensor, y_test_tensor)

        train_and_val_loader = DataLoader(dataset=train_and_val_data_set, batch_size=Constants.train_batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data_set, batch_size=Constants.val_batch_size,shuffle=True)
        test_loader = DataLoader(dataset=test_data_set, batch_size=Constants.test_batch_size,shuffle=True)

        stop=False
        epoch=0
        best_f1=0
        best_f1_list=[]

        while not stop:
            epoch+=1
            print("\nTraining epoch number {epoch}\n".format(epoch=epoch))
            """Train the model on the training set"""
            for x_train_and_val_batch, y_train_and_val_batch in train_and_val_loader:
                train_step(x_train_and_val_batch, y_train_and_val_batch)
            """Evaluate the model after it finished the epoch"""
            with torch.no_grad():
                flared_learning_model.eval()
                probs_pred = flared_learning_model.predict_prob(x_train_and_val_tensor).detach().numpy()
                active_probs = probs_pred[:, 1]
                precision, recall, thresholds = precision_recall_curve(y_train_and_val_tensor, active_probs)
                best_threshold,_ = Clustering.best_threshold(precision, recall, thresholds)

                y_val_pred=flared_learning_model.predict(flared_learning_model(x_val_tensor), best_threshold)
                f1_score=precision_recall_fscore_support(y_val_tensor, y_val_pred, average='binary')[2]
                best_f1_list.append(f1_score)
                if f1_score>best_f1:
                    best_f1=f1_score
                    torch.save({
                        'model_state_dict': flared_learning_model.state_dict(),
                        'precision':precision,
                        'recall': recall,
                        'best_threshold':best_threshold
                    },os.path.join(script_dir,Constants.MODEL_PATH))
                stop=Clustering.early_stopping(best_f1_list,patience=20)

        """Loss visualization through different epochs"""
        fig,axes=plt.subplots(1,4)
        axes[0].plot(range(1,epoch+1),best_f1_list,label='Validation_f1',ls='--')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Validation F1')
        axes[0].set_title('Active cases prediction,\n Hidden={hidden}\n Lr={lr}\n Epochs={ep}'.format(hidden=Constants.hidden_size,lr=Constants.lr,ep=epoch))
        axes[0].legend()

        check_point=torch.load(os.path.join(script_dir,Constants.MODEL_PATH))
        recall=check_point['recall']
        precision=check_point['precision']
        """precision-recall curve"""
        axes[1].plot(recall,precision)
        axes[1].set_ylabel('Precision')
        axes[1].set_xlabel('Recall')
        axes[1].set_title('Precision-Recall curve on train and validation')

        trained_model = Clustering.learning_model(Constants.nn_structure, Constants.output_layer_size)
        trained_model.load_state_dict(check_point['model_state_dict'])
        trained_model.eval()
        best_threshold=check_point['best_threshold']
        """Confusion matrix"""

        cm_train = confusion_matrix(y_train_and_val_tensor, trained_model.predict(trained_model(x_train_and_val_tensor),best_threshold))
        cm_test = confusion_matrix(y_test_tensor, trained_model.predict(trained_model(x_test_tensor),best_threshold))
        sns.heatmap(cm_train,annot=True,ax=axes[2],cmap="Blues",cbar=False,fmt="d")
        sns.heatmap(cm_test,annot=True,ax=axes[3],cmap="Blues",cbar=False,fmt="d")
        axes[2].set_ylabel('True label')
        axes[2].set_xlabel('Predicted label')
        axes[2].set_title('Train confusion matrix')
        axes[3].set_ylabel('True label')
        axes[3].set_xlabel('Predicted label')
        axes[3].set_title('Test confusion matrix')

        plt.tight_layout()
        plt.show()
        plt.close()
        "Rest the model towards the next division of the data "
        flared_learning_model.apply(Clustering.weight_reset)





else:
    raise Exception('The target feature {target} is not supported'.format(target=target_feature))
