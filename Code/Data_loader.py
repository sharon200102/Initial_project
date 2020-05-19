import pandas as pd
import requests
import os
# because the exported features file was in a txt format, I had difficulties in reading it,so some transformations were made.
def data_loader_exported_features(column_page_url,values_page_url,column_delimiter='\t',value_delimiter='\t'):

    column_page = requests.get(column_page_url)
    column_info=column_page.text
    column_name_list=column_info.split(column_delimiter)
    exported_features=pd.read_csv(values_page_url,header=None,names=column_name_list,delimiter=value_delimiter)
    return exported_features

def files_read_csv_fails_to_data_frame(file_path, amount_of_column_name_rows_in_file=1, columns_sep='\t', data_sep='\t', decoder="utf-8"):
  column_names=[]
  data_rows=[]
  if(os.path.isfile(file_path)):
      f=open(file_path)
      for line_number, line in enumerate(f):
          if line_number<amount_of_column_name_rows_in_file:
              column_names.extend(line.split(columns_sep))
          else:
              data_rows.append(line.split(data_sep))
  else:
        f_page=requests.get(file_path)
        for line_number, line in enumerate(f_page.iter_lines()):
            line = line.decode(decoder)
            if line_number < amount_of_column_name_rows_in_file:
                column_names.extend(line.split(columns_sep))
            else:
                data_rows.append(line.split(data_sep))

  df=pd.DataFrame(data_rows,columns=column_names)
  return df