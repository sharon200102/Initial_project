import itertools
import matplotlib
import pickle
from scipy.stats import spearmanr
import random
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import os
from os.path import join
from statsmodels.stats.multitest import multipletests
import sys
from Plot.plot_bacteria_intraction_network import plot_bacteria_intraction_network
plt.rcParams["image.cmap"] = "Set1"

"""
             Iterate through all n chose two sub groups of columns in the data, and plot the relationship between them.
             colored by the categorical column inserted.
             Example-https://github.com/sharon200102/Initial_project/blob/ICA_on_the_whole_data/Graphs/All%20timepoints/Graphs%20after%20Log/Data_after_ICA_relationship_between_features_.png
             
            Parameters:
            dataframe - An ordinary DataFrame that its columns will be plotted.
            folder - A  String that represents the name of the folder where the plot will be stored (creates one if the folder doesn't exist) 
            color - Series of discrete values which will be used to color each and every plot (Defult None).
            Title - String title for the plot (Default Generic title).
            labels_dict - A dictionary which maps every discrete color value into a string, for more informal legend (Default None).   
            figure size -  A tuple represents the plot figure size (default generic size).
            other size variables- Its possible to change x/y/legend/title sizes.
            kwargs - Parameters inserted  to each scatter plot. 
        Returns The created figure.
        
"""


def relationship_between_features(dataframe, folder, color=None, title="Relationship_between_features",title_size=30,
                                  labels_dict=None, figure_size=(18, 18),axis_labels_size=15,legend_size=15, **kwargs):
    number_of_columns = dataframe.shape[1]
    fig, axes = plt.subplots(number_of_columns, number_of_columns, squeeze=False, figsize=figure_size)
    if color is not None:
        groups = dataframe.groupby(color)
    # Iterate through all n chose two subgroups where n is the number of columns.

    for row, col in list(itertools.combinations(list(range(0, number_of_columns)), 2)):
        if color is not None:
            for name, group in groups:
                if labels_dict is not None:
                    axes[row][col].scatter(group.iloc[:, row], group.iloc[:, col], label=labels_dict[name], **kwargs)
                else:
                    axes[row][col].scatter(group.iloc[:, row], group.iloc[:, col], label=name, **kwargs)

        else:
            axes[row][col].scatter(dataframe.iloc[:, row], dataframe.iloc[:, col], **kwargs)

            # Shape the subplot to be in a  triangular form and set titles.
        fig.delaxes(axes[col][row])
        axes[row][col].set_xlabel(dataframe.columns[row], fontsize=axis_labels_size)
        axes[row][col].set_ylabel(dataframe.columns[col],fontsize=axis_labels_size)
    for diag in range(0, number_of_columns):
        fig.delaxes(axes[diag][diag])

    # Set a main title and save the figure.
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels,fontsize=legend_size)
    fig.suptitle(title,size= title_size)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)

    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(join(folder, title.replace(" ", "_").replace("\n", "_") + ".png"))

    plt.close()
    return fig


"""
A class that reproduces a plot of the progress in time of all columns attribute mean.
T-test will be performed between all pairs of groups (component split by a specific timepoint->split by a binary attribute->perform ttest between the groups)
example-https://github.com/sharon200102/Initial_project/blob/ICA_on_the_whole_data/Graphs/All%20timepoints/Graphs%20after%20Log/Progress_in_time_of_column_attribute_mean_ICA.png

The constructor must receive:
    dataframe - An ordinary DataFrame (n_samples,n_components) that its columns mean progress will be plotted.
    time_series- A series object (n_samples,) that represents the time of each row in the dataframe.

Optional Arguments:
    attribute_series- Additionally to time_series, the progress of each column will be also split by an attribute_series.
    (attribute must be binary)
    Plot arguments:
        The user can edit the plot defaults by inserting
        line_styles - array of matplotlib line styles.
        markers - array of matplotlib markers.
        colors- array of matplotlib colors
        figure_size - A tuple for the figure size.
        fontsize - for the legend font size.
        
  

 
"""
class progress_in_time_of_column_attribute_mean(object):
    def __init__(self,dataframe,time_series,attribute_series=None,**kwargs):

        self.dataframe=dataframe
        self.time_series=time_series
        self.attribute_series=attribute_series
        self._pvalues_matrix= None if attribute_series is None else self._create_p_values_matrix()
        self._component_mean=self._component_factorization()
        self.asterisk_matrix=self._pvalues_matrix.applymap(self._transform_p_to_marker)

        self.line_styles=kwargs.get('line_styles',['solid','dashed'])
        self.markers=kwargs.get('markers',range(2,12))
        self.colors=kwargs.get('colors',['b','g','r','c','m'])
        figure_size=kwargs.get('figure_size',(10,15))
        self.fig=plt.figure(figsize=figure_size)
        self.new_plot=self.fig.add_subplot(111)
        self.margin=kwargs.get('margin',0.006)
        self.fontsize=kwargs.get('fontsize',7)



    """The function factorizes the dataframe into a list, in the list every component is mapped to a binary tuple.
        both arguments are lists that consist the means in all time points, each one for a different group.
        [([component0_mean_in_tp0_group0,...],[component0_mean_in_tp0_group1,...])*n_components]
    """
    def _component_factorization(self):
        all_componets=[]
        component_mean_first_group=[]
        component_mean_second_group=[]
        for col in self.dataframe.columns:
            relevant_feature=self.dataframe[col]
            for time_point in sorted(self.time_series.unique()):
                relevant_feature_in_specific_time=relevant_feature[self.time_series==time_point]
                if self.attribute_series is None:
                    component_mean_first_group.append(relevant_feature_in_specific_time.mean())
                else:
                    attribute_first_group=relevant_feature_in_specific_time[self.attribute_series==sorted(self.attribute_series.unique())[0]]
                    attribute_second_group=relevant_feature_in_specific_time[self.attribute_series==sorted(self.attribute_series.unique())[1]]
                    component_mean_first_group.append(attribute_first_group.mean())
                    component_mean_second_group.append(attribute_second_group.mean())

            if not component_mean_second_group:
                all_componets.append((component_mean_first_group))
            else:
                all_componets.append((component_mean_first_group,component_mean_second_group))

            component_mean_first_group = []
            component_mean_second_group = []
        return  all_componets

    """ Creates a p_value dataframe with a (n_components,unique time points) shape.
       The value in the (x,y) coordinate represents the p-value of the ttest performed on the groups of component x in time y.  
    """
    def _create_p_values_matrix(self):
        pvalues_matrix=pd.DataFrame(0.0, index=self.dataframe.columns, columns=list(map(lambda x: str(x), sorted(self.time_series.unique()))))
        for col in self.dataframe.columns:
            relevant_feature=self.dataframe[col]
            for time_point in sorted(self.time_series.unique()):
                relevant_feature_in_specific_time=relevant_feature[self.time_series==time_point]
                attribute_first_group = relevant_feature_in_specific_time[self.attribute_series == sorted(self.attribute_series.unique())[0]]
                attribute_second_group = relevant_feature_in_specific_time[self.attribute_series == sorted(self.attribute_series.unique())[1]]
                groups_p_val = stats.ttest_ind(attribute_first_group, attribute_second_group, equal_var=False)[1]
                pvalues_matrix.at[col, str(time_point)] = groups_p_val
        return pvalues_matrix

    """Transforms a pvalue to asterisks"""
    @staticmethod
    def _transform_p_to_marker(p_val):
        if 0.01<p_val<=0.05:
            return '*'
        elif 0.001<p_val<=0.01:
            return '**'
        elif p_val<=0.001:
            return '***'
        else:
            return None

    """Adds the mean progress lines to the plot"""
    def _add_lines(self):
        for component,component_name,color,marker in zip(self._component_mean,self.dataframe.columns,self.colors,self.markers):
            if self.attribute_series is None:
                for group,line_style in zip(component,self.line_styles):
                    self.new_plot.plot(sorted(self.time_series.unique()),group,color=color,marker=marker,linestyle=line_style,label=component_name)
            else:
                for group,attribute_val,line_style in zip(component,sorted(self.attribute_series.unique()),self.line_styles):
                    self.new_plot.plot(sorted(self.time_series.unique()),group,color=color,marker=marker,linestyle=line_style,label="{0} {1}".format(component_name,attribute_val))

    """Adds the ttest p-value asterisks to the plot"""
    def _add_asterisks(self):
        bottom_lim,top_lim=self.new_plot.get_ylim()
        total_margin=0
        for col in self.asterisk_matrix.columns:
            total_margin = 0
            for asterisks,color in zip (self.asterisk_matrix[col],self.colors):
                total_margin += self.margin
                if asterisks is not None:
                    self.new_plot.annotate(asterisks,(float(col),bottom_lim-total_margin),color=color)
        self.new_plot.set_ylim(bottom=bottom_lim-total_margin,top=top_lim+total_margin)

    def plot(self):
        self.new_plot.set_xticks(sorted(self.time_series.unique()))
        self._add_lines()
        if self.attribute_series is not None:
            self._add_asterisks()
        plt.legend(fontsize=self.fontsize)
        return self.new_plot




class bacteria_intraction_in_time(object):
    def __init__(self,dataframe,key_columns_names,time_feature_name,binary_colors=['#ff0000','	#0eff00'],delimiter=None):
        self.dataframe=dataframe
        self.time_feature_name=time_feature_name
        self.key_columns_names=key_columns_names
        self.groups=dataframe.groupby(self.key_columns_names)
        self.relevant_columns=list(self.dataframe.drop(self.key_columns_names,axis=1).columns)
        self.delta_dataframe,self.feature_value_dataframe=self._group_features_development_in_subsequent_times()
        self.correlation_dataframe,self.p_value_dict=self._correlation_relationship()
        self.nodes,self.edges=self._remove_unconnected_nodes(self._significant_pvals_to_edges())
        self.numeric_edges=[(self.nodes.index(x),self.nodes.index(y)) for x,y in self.edges]
        self.edges_colors=self._adjust_colors(binary_colors)
        self.nodes=self.nodes if delimiter is not None else [self._cut_node_names(name,delimiter) for name in self.nodes ]
        self.numeric_nodes=list(range(0,len(self.nodes)))

    def _subsequent_times(self,group):
        time_list=sorted(list(group[self.time_feature_name]))
        return [(current_time,next_time)for current_time,next_time in zip(time_list,time_list[1:]) if current_time+1==next_time]

    def _group_features_development_in_subsequent_times(self):
        delta_list=[]
        value_list=[]
        for name,group in self.groups:
            group=group.drop(self.key_columns_names,axis=1)
            for first_time,second_time in self._subsequent_times(group):
                first_row=group[group[self.time_feature_name]==first_time].squeeze()
                second_row=group[group[self.time_feature_name]==second_time].squeeze()
                delta_series=second_row.subtract(first_row)
                delta_list.append(list(delta_series))
                value_list.append(list(first_row))
        return  pd.DataFrame(delta_list,columns=self.relevant_columns),pd.DataFrame(value_list,columns=self.relevant_columns)
   
   
    def _correlation_relationship(self):
        correlation_dataframe=pd.DataFrame(0.0,index=self.feature_value_dataframe.columns,columns=self.delta_dataframe.columns)
        p_value_dict={}
        for delta_col in self.delta_dataframe.columns:
            for feature_col in self.feature_value_dataframe.columns:
                correlation, pvalue = spearmanr(self.feature_value_dataframe[feature_col], self.delta_dataframe[delta_col],nan_policy='omit')
                if pd.isnull(correlation):
                    correlation=0
                    pvalue=1
                correlation_dataframe.at[feature_col,delta_col]=correlation
                p_value_dict[(feature_col,delta_col)]=pvalue
        return correlation_dataframe,p_value_dict

    def _significant_pvals_to_edges(self):
        significant_pvals=multipletests(list(self.p_value_dict.values()))[0]
        edges=[edge for i,edge in enumerate(self.p_value_dict.keys()) if significant_pvals[i]]
        return edges

    def _remove_unconnected_nodes(self,significant_edges):

        nodes=[]
        edges=significant_edges.copy()
        for possible_node in self.relevant_columns:
            self_edge=(possible_node,possible_node)
            edges_related_to_node=list(filter(lambda x: True if x[0]==possible_node or x[1]==possible_node else False,edges))
            if len(edges_related_to_node)==1 and edges_related_to_node[0]==(self_edge):
                edges.remove(self_edge)
            elif len(edges_related_to_node)>=1:
                nodes.append(possible_node)
        return nodes,edges



    def _adjust_colors(self,binary_colors):
        return list(map(lambda edge: binary_colors[0] if self.correlation_dataframe.at[edge[0],edge[1]]>0 else binary_colors[1],self.edges))

    @staticmethod
    def _cut_node_names(name,delimiter):
            return name.split(delimiter)[-1]

    def plot(self,**kwargs):
        plot_bacteria_intraction_network(bacteria=self.nodes,node_list=self.numeric_nodes,edge_list=self.numeric_edges,color_list=self.edges_colors,**kwargs)

    def export_edges_to_csv(self,name_of_file):
        edges_dataframe=pd.DataFrame(self.edges,columns=['Influential feature' ,'Affected feature'])
        edges_dataframe.to_csv('{file}.csv'.format(file=name_of_file))





def draw_rhos_calculation_figure(id_to_binary_tag_map, preproccessed_data, title, taxnomy_level, num_of_mixtures=10,
                                 ids_list=None, save_folder=None):
    import matplotlib.pyplot as plt

    # calc ro for x=all samples values for each bacteria and y=all samples tags
    features_by_bacteria = []
    if ids_list:
        ids_list = [i for i in ids_list if i in preproccessed_data.index]
        X = preproccessed_data.loc[ids_list]
        y = [id_to_binary_tag_map[id] for id in ids_list]

    else:
        x_y = [[preproccessed_data.loc[key], val] for key, val in id_to_binary_tag_map.items()]
        X = pd.DataFrame([tag[0] for tag in x_y])
        y = [tag[1] for tag in x_y]

    # remove samples with nan as their tag
    not_nan_idxs = [i for i, y_ in enumerate(y) if str(y_) != "nan"]
    y = [y_ for i, y_ in enumerate(y) if i in not_nan_idxs]
    X = X.iloc[not_nan_idxs]



    mixed_y_list = []
    for num in range(num_of_mixtures):  # run a couple time to avoid accidental results
        mixed_y = y.copy()
        random.shuffle(mixed_y)
        mixed_y_list.append(mixed_y)

    bacterias = X.columns
    real_rhos = []
    real_pvalues = []
    used_bacterias = []
    mixed_rhos = []
    mixed_pvalues = []

    bacterias_to_dump = []
    for i, bact in enumerate(bacterias):
        f = X[bact]
        num_of_different_values = set(f)
        if len(num_of_different_values) < 2:
            bacterias_to_dump.append(bact)
        else:
            features_by_bacteria.append(f)
            used_bacterias.append(bact)

            rho, pvalue = spearmanr(f,y,nan_policy='omit')
            if str(rho) == "nan":
                print(bact)
            real_rhos.append(rho)
            real_pvalues.append(pvalue)

            for mix_y in mixed_y_list:
                rho_, pvalue_ = spearmanr(f, mix_y,nan_policy='omit')
                mixed_rhos.append(rho_)
                mixed_pvalues.append(pvalue_)

    print("number of bacterias to dump: " + str(len(bacterias_to_dump)))
    print("percent of bacterias to dump: " + str(len(bacterias_to_dump)/len(bacterias) * 100) + "%")

    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags
    real_min_rho = min(real_rhos)
    real_max_rho = max(real_rhos)
    mix_min_rho = min(mixed_rhos)
    mix_max_rho = max(mixed_rhos)

    real_rho_range = real_max_rho - real_min_rho
    mix_rho_range = mix_max_rho - mix_min_rho

    # new method - all the items out of the mix range + 1% from the edge of the mix
    upper_bound = np.percentile(mixed_rhos, 99)
    lower_bound = np.percentile(mixed_rhos, 1)

    significant_bacteria_and_rhos = []
    for i, bact in enumerate(used_bacterias):
        if real_rhos[i] < lower_bound or real_rhos[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, real_rhos[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])
    if save_folder:
        with open(join(save_folder, "significant_bacteria_" + title + "_taxnomy_level_" + str(taxnomy_level)
                                    + "_.csv"), "w") as file:
            file.write("rho,bact\n")
            for s in significant_bacteria_and_rhos:
                file.write(str(s[1]) + "," + str(s[0]) + "\n")
                # דפנההה מה קורה?? איך עובר היום?
    # draw the distribution of real rhos vs. mixed rhos
    # old plots
    [count, bins] = np.histogram(mixed_rhos, 50)
    # divide by 'num_of_mixtures' fo avoid high number of occurrences due to multiple runs for each mixture
    plt.bar(bins[:-1], count/num_of_mixtures, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="mixed tags",
            color="#d95f0e")
    [count, bins2] = np.histogram(real_rhos, 50)
    plt.bar(bins2[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.8, label="real tags", color="#43a2ca")
    # plt.hist(real_rhos, rwidth=0.6, bins=50, label="real tags", color="#43a2ca" )
    # plt.hist(mixed_rhos, rwidth=0.9, bins=50, alpha=0.5, label="mixed tags", color="#d95f0e")
    plt.title("Real tags vs. Mixed tags at " + title.replace("_", " "))
    plt.xlabel('Rho value')
    plt.ylabel('Number of bacteria')
    plt.legend()
    # print("Real tags_vs_Mixed_tags_at_" + title + "_combined.png")
    # plt.show()
    if save_folder:
        plt.savefig(join(save_folder, "Real_tags_vs_Mixed_tags_at_" + title.replace(" ", "_")
                         + "_taxnomy_level_" + str(taxnomy_level) + ".svg"), bbox_inches='tight', format='svg')
    plt.close()

    # positive negative figures
    bacterias = [s[0] for s in significant_bacteria_and_rhos]
    real_rhos = [s[1] for s in significant_bacteria_and_rhos]
    # extract the last meaningful name - long multi level names to the lowest level definition

    short_bacterias_names = []
    for f in bacterias:
        i = 1
        while len(f.split(";")[-i]) < 5 or f.split(";")[-i] == 'Unassigned':  # meaningless name
            i += 1
            if i > len(f.split(";")):
                i -= 1
                break
        short_bacterias_names.append(f.split(";")[-i])
    # remove "k_bacteria" and "Unassigned" samples - irrelevant
    k_bact_idx = []
    for i, bact in enumerate(short_bacterias_names):
        if bact == 'k__Bacteria' or bact == 'Unassigned':
            k_bact_idx.append(i)

    if k_bact_idx:
        [short_bacterias_names, real_rhos, bacterias] = pop_idx(k_bact_idx, [short_bacterias_names, real_rhos, bacterias])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(bacterias))
    coeff_color = []
    for x in real_rhos:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    ax.barh(y_pos, real_rhos, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_bacterias_names)
    plt.yticks(fontsize=7)
    plt.title(title.replace("_", " "))
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    if save_folder:
        plt.savefig(join(save_folder, "pos_neg_correlation_at_" + title.replace(" ", "_")
                         + "_taxnomy_level_" + str(taxnomy_level) + ".svg"), bbox_inches='tight', format='svg')
    plt.close()
