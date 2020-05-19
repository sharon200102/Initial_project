import pandas as pd
import requests

# because the exported features file was in a txt format, I had difficulties in reading it,so some transformations were made.
def data_loader_exported_features(column_page_url,values_page_url,column_delimiter='\t',value_delimiter='\t'):

    column_page = requests.get(column_page_url)
    column_info=column_page.text
    column_name_list=column_info.split(column_delimiter)
    exported_features=pd.read_csv(values_page_url,header=None,names=column_name_list,delimiter=value_delimiter)
    return exported_features

def taxonomy(taxonomy_page_url,sep='\t'):
  firstline=True
  Feature_ID_list=[]
  Taxon_list=[]
  taxonomy_page=requests.get(taxonomy_page_url)
  for line in taxonomy_page.iter_lines():
      if firstline==False:
        line=line.decode("utf-8")
        id_taxon_con=line.split(sep)
        Feature_ID_list.append(id_taxon_con[0])
        Taxon_list.append(id_taxon_con[1])
      else:
        firstline=False
        line = line.decode("utf-8")
        names= line.split(sep)
  df=pd.DataFrame({names[0]:Feature_ID_list,names[1]:Taxon_list})
  return df