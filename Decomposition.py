import pandas as pd

def decompose(data,dec_func,n_components=2,**kwargs):
    dec_obj = dec_func(n_components=n_components,**kwargs)
    dec_obj.fit(data)
    components = dec_obj.fit_transform(data)
    #changing the columns names.
    componentsDF = pd.DataFrame(data=components, columns=list(map(lambda num:'component '+str(num),range(1,n_components+1))))
    #visualization
    return dec_obj,componentsDF
