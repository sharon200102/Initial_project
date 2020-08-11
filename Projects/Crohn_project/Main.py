import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
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

    tensor_data = torch.from_numpy(dec_data_adusted_to_target.to_numpy()).type(torch.FloatTensor)
    tensor_target = torch.from_numpy(target_df['Tag'].to_numpy()).type(torch.LongTensor)


    flared_learning_model=Clustering.learning_model(Constants.nn_structure,Constants.output_layer_size)
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(flared_learning_model.parameters(), lr=Constants.lr)
    train_step=Clustering.make_train_step(flared_learning_model,loss_fn,optimizer)




    train_idx_list, test_idx_list = train_test_split(dec_data_adusted_to_target, target_df['Tag'], target_df['patient_No'],return_itr=False,
                                                     random_state=1)



    for current_train_idx,current_test_idx in zip(train_idx_list,test_idx_list):
        x_train_tensor = tensor_data[current_train_idx]
        y_train_tensor = tensor_target[current_train_idx]
        x_test_tensor = tensor_data[current_test_idx]
        y_test_tensor = tensor_target[current_test_idx]

        train_data_set = TensorDataset(x_train_tensor, y_train_tensor)
        test_data_set = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset=train_data_set, batch_size=Constants.train_batch_size,shuffle=True)
        test_loader = DataLoader(dataset=test_data_set, batch_size=Constants.test_batch_size,shuffle=True)
        train_average_loss=[]
        test_average_loss=[]

        for epoch in range(Constants.epochs):
            """Train the model on the training set"""
            for x_train_batch, y_train_batch in train_loader:
                train_step(x_train_batch, y_train_batch)
            """Evaluate the model after it finished the epoch"""
            sum_loss=0
            with torch.no_grad():
                flared_learning_model.eval()
                predictions_on_train=[]
                real_y_train=[]
                for x_train_batch, y_train_batch in train_loader:
                    yhat = flared_learning_model(x_train_batch)
                    sum_loss += loss_fn(yhat, y_train_batch).item()
                    real_y_train.extend(y_train_batch)
                    predictions_on_train.extend(flared_learning_model.predict(yhat))

                train_average_loss.append(sum_loss/len(train_loader.dataset))
                sum_loss=0
                predictions_on_test = []
                real_y_test = []

                for x_test_batch, y_test_batch in test_loader:
                    yhat = flared_learning_model(x_test_batch)
                    sum_loss+=loss_fn(yhat,y_test_batch).item()
                    real_y_test.extend(y_test_batch)
                    predictions_on_test.extend(flared_learning_model.predict(yhat))

                test_average_loss.append(sum_loss/len(test_loader.dataset))

        """Loss visualization through different epochs"""
        fig,axes=plt.subplots(1,3)
        axes[0].plot(range(1,Constants.epochs+1),train_average_loss,label='Train_loss')
        axes[0].plot(range(1,Constants.epochs+1),test_average_loss,label='Test_loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Average loss')
        axes[0].set_title('Active cases prediction,\n Hidden={hidden}\n Lr={lr}\n Epochs={ep}'.format(hidden=Constants.hidden_size,lr=Constants.lr,ep=Constants.epochs))

        """Confusion matrix"""
        axes[0].legend()
        cm_train = confusion_matrix(real_y_train, predictions_on_train)
        cm_test = confusion_matrix(real_y_test, predictions_on_test)
        sns.heatmap(cm_train,annot=True,ax=axes[1],cmap="Blues",cbar=False,fmt="d")
        sns.heatmap(cm_test,annot=True,ax=axes[2],cmap="Blues",cbar=False,fmt="d")
        axes[1].set_ylabel('True label')
        axes[1].set_xlabel('Predicted label')
        axes[1].set_title('Train confusion matrix')
        axes[2].set_ylabel('True label')
        axes[2].set_xlabel('Predicted label')
        axes[2].set_title('Test confusion matrix')

        plt.tight_layout()
        plt.show()
        plt.close()
        "Rest the model towards the next division of the data "
        flared_learning_model.apply(Clustering.weight_reset)





else:
    raise Exception('The target feature {target} is not supported'.format(target=target_feature))
