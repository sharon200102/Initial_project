from Code.Preprocessing_actions import Normalization
from sklearn.decomposition import PCA, FastICA
column_page_url = 'https://raw.githubusercontent.com/sharon200102/Initial_project/master/Projects/CRC_data/data/column_names_of_exported_feature-table_for_YoramL.txt'
values_page_url='https://raw.githubusercontent.com/sharon200102/Initial_project/master/Projects/CRC_data/data/values_of_exported_feature-table_for_YoramL.txt'
maping_page_url='https://raw.githubusercontent.com/sharon200102/Initial_project/master/Projects/CRC_data/data/mapping%20file%20with%20data%20Baniyahs%20Merge.csv'
taxonomy_page_url='https://raw.githubusercontent.com/sharon200102/Initial_project/master/Projects/CRC_data/data/taxonomy.tsv'
column_page_delimiter='\t'
value_page_delimiter='\t'
relevant_categorical_names=['Group','Treatment','TimePointNum']
immune_system_features=['spleen_weight','cell_spleen','MDSC_GR1_spleen','MFI_zeta_spleen','cell_BM','MDSC_GR1_bm']
mapping_and_general_info_columns=['BarcodeSequence','LinkerPrimerSequence','ReversePrimer','plate','Experiment','Genotype','MiceNum','CageNum','Mouse','Treatment','SampleType','DayOfSam','TimePoint','TimePointNum','TimeGroup','tumor_load','spleen_weight','cell_spleen','MDSC_GR1_spleen','MFI_zeta_spleen','cell_BM','MDSC_GR1_bm','Project','Description','WellPosition','GroupTreat','Group','Run']
mapping_and_general_info_columns_no_identification=['#SampleID','BarcodeSequence','LinkerPrimerSequence','ReversePrimer','plate','Genotype','Mouse','Treatment','SampleType','DayOfSam','TimePoint','TimeGroup','tumor_load','spleen_weight','cell_spleen','MDSC_GR1_spleen','MFI_zeta_spleen','cell_BM','MDSC_GR1_bm','Project','Description','WellPosition','GroupTreat','Group','Run']
identification_columns=['Experiment','MiceNum','CageNum']
THRESHOLD=0.8
P_VALUE_THRESHOLD=0.05
LEVEL_TO_REDUCE=6
TimePoints_list=['All','0']
prediction_sort=['Binary','Multi','Regression']
Evaluation_name_list=['Accuracy','Auc','R2']
normalization_dict={'Robust':Normalization.Robust_std,'Zscore':Normalization.zscore_norm,'MinMax':Normalization.minmax_norm,'Log':Normalization.log_normalization}
normalization_parameters_dict={'Robust':(),'Zscore':(),'MinMax':(),'Log':[1]}
dimension_reduction_dict={'PCA':PCA,'ICA':FastICA}
N_SPLITS=3
MAX_NIGH=7
