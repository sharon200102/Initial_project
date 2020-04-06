from Preprocessing_actions import Normalization
from sklearn.decomposition import PCA, FastICA
column_page_url = 'https://raw.githubusercontent.com/sharon200102/Initial_project/master/column_names_of_exported_feature-table_for_YoramL.txt'
values_page_url='https://raw.githubusercontent.com/sharon200102/Initial_project/master/values_of_exported_feature-table_for_YoramL.txt'
maping_page_url='https://raw.githubusercontent.com/sharon200102/Initial_project/master/mapping%20file%20with%20data%20Baniyahs%20Merge.csv'
taxonomy_page_url='https://raw.githubusercontent.com/sharon200102/Initial_project/master/taxonomy.tsv'
column_page_delimiter='\t'
value_page_delimiter='\t'
relevant_categorical_names=['Group','Treatment','TimePointNum']
columns_to_be_dropped=['BarcodeSequence','LinkerPrimerSequence','ReversePrimer','plate','Experiment','Genotype','MiceNum','CageNum','Mouse','Treatment','SampleType','DayOfSam','TimePoint','TimePointNum','TimeGroup','tumor_load','spleen_weight','cell_spleen','MDSC_GR1_spleen','MFI_zeta_spleen','cell_BM','MDSC_GR1_bm','Project','Description','WellPosition','GroupTreat','Group','Run']
THRESHOLD=0.8
normalization_dict={'Robust':Normalization.Robust_std,'Zscore':Normalization.zscore_norm,'MinMax':Normalization.minmax_norm}
dimension_reduction_dict={'PCA':PCA,'ICA':FastICA}
N_SPLITS=3
MAX_NIGH=7
