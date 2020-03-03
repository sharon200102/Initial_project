import pandas as pd
"""
column_file_name=r'C:\sharon\second_degree\Initial_project\column_names_of_exported_feature-table_for_YoramL.txt'
values_file_name=r'C:\sharon\second_degree\Initial_project\values_of_exported_feature-table_for_YoramL.txt'
maping_file_name=r'C:\sharon\second_degree\Initial_project\mapping file with data Baniyahs Merge.csv'
"""
def data_loader_exported_features(column_file_name,values_file_name,column_delimiter,value_delimiter):

    col_file=open(column_file_name)
    column_info=col_file.read()
    column_name_list=column_info.split(column_delimiter)
    exported_features=pd.read_csv(values_file_name,header=None,names=column_name_list,delimiter=value_delimiter)
    return exported_features

def data_rearrangement(data):
    data.set_index('#OTU_ID',inplace=True)
    data=data.T
    data.insert(loc=0,column='#SampleID',value=data.index)
    data.reset_index(drop=True,inplace=True)
    data=data.rename_axis([None], axis=1).rename_axis('index')
    return data

"""
samples_results=data_rearrangement(data_loader_exported_features(column_file_name,values_file_name,'\t','\t'))
mapping_table=pd.read_csv(maping_file_name)
print(pd.merge(samples_results,mapping_table,on='#SampleID'))
"""