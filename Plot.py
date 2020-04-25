import itertools
import seaborn as sns;
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import Constants
from math import log10
import os
from os.path import join
sns.set()
plt.rcParams["image.cmap"] = "Set1"

"""
             Iterate through all n chose two sub groups of columns in the data, and plot the relationship between them.
             colored by the categorical column inserted.
             
            Parameters:
            dataframe - An ordinary DataFrame that its columns will be plotted.
            folder - A  String that represents the name of the folder where the plot will be stored (creates one if the folder doesn't exist) 
            color - Series of discrete values which will be used to color each and every plot (Defult None).
            Title - String title for the plot (Default Generic title).
            labels_dict - A dictionary which maps every discrete color value into a string, for more informal legend (Default None).   
            figure size -  A tuple represents the plot figure size (default generic size).
            kwargs - Parameters inserted  to each scatter plot. 
"""


def relationship_between_features(dataframe, folder, color=None, title="Relationship_between_features",
                                  labels_dict=None, figure_size=(18, 18), **kwargs):
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
        axes[row][col].set_xlabel(dataframe.columns[row])
        axes[row][col].set_ylabel(dataframe.columns[col])
    for diag in range(0, number_of_columns):
        fig.delaxes(axes[diag][diag])

    # Set a main title and save the figure.
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(title)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)

    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(join(folder, title.replace(" ", "_").replace("\n", "_") + ".png"))

    plt.close()
    return fig


"""
At the moment: the function plots the progress of the mean of each column,
based on the unique values of the categorical column inserted. 
"""


def progress_in_time_of_column_attribute_mean(dataframe, time_series,folder,
                                              attribute_series=None,
                                              splitter_name="",
                                              title='Progress_in_time_of_column_attribute_mean',figure_size=(10,10)):
    # the marker will be responsible to the difference between the columns.
    cmap = matplotlib.cm.get_cmap()
    lines_array = list(matplotlib.lines.lineStyles.keys())
    marker_counter = 0
    fig = plt.figure(figsize=figure_size)
    new_plot = fig.add_subplot(111)
    # if there is no splitter
    if attribute_series is None:

        # Add the categorical column to the data.
        dataframe = dataframe.assign(time_point=time_series.values)
        # Group the data based on the categorical column.
        grouped = dataframe.groupby(['time_point']).mean()
        # Every column in grouped contains the mean of the column for every unique value in categorical_series
        for col_name in grouped.columns:
            # Plot the line the goes through the means of the column in the different categorical values.
            new_plot.plot(grouped.index, grouped[col_name], marker=marker_counter, markersize=10,
                          label=col_name)
            marker_counter += 1

        # Plot A vertical line for each unique categorical value.

        for time_point in grouped.index:
            ymin = grouped.loc[time_point].min()
            ymax = grouped.loc[time_point].max()
            new_plot.plot([time_point, time_point], [ymin, ymax], c='k')
    # Splitter exists
    else:
        # Add both of the categorical columns
        extended_dataframe = dataframe.assign(time_point=time_series.values, attribute=attribute_series)
        # Groupby both columns and make only time as index avoiding two indices situation.
        grouped = extended_dataframe.groupby(['time_point', 'attribute']).mean().reset_index(level=1)
        # save and remove the splitter because its not truly a part of the data,it was only added because of the groupby
        attribute_col = grouped['attribute']
        grouped.drop('attribute', axis=1, inplace=True)

        #   The columns in grouped represent the progress of the original column over categorical.

        for color_index, col_name in enumerate(grouped.columns):
            # Select every column by all possible splitter values and plot the progress.
            color=cmap(color_index/len(grouped.columns))
            for line_style_index, splitter_val in enumerate(attribute_col.unique()):
                linestyle =lines_array[line_style_index]
                splitter_progress = grouped[col_name][attribute_col == splitter_val]
                new_plot.plot(splitter_progress.index, splitter_progress, color=color, linestyle =linestyle, label=col_name + "_" + str(splitter_val))

        # Plot A vertical line for each unique categorical value.
        for time_point in grouped.index.unique():
            grouped_in_specific_time_point = grouped[grouped.index == time_point]
            ymax = max(grouped_in_specific_time_point.max(axis=1))
            ymin = min(grouped_in_specific_time_point.min(axis=1))
            new_plot.plot([time_point, time_point], [ymin, ymax], c='k')

    new_plot.set_xlabel('Time')
    new_plot.set_ylabel('Mean value')
    new_plot.set_title(title)
    plt.legend(title="Splitted by: " + splitter_name)
    plt.tight_layout()
    plt.close()


    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(join(folder, title.replace(" ", "_").replace("\n", "_") + ".png"))
    return fig


"""
The function first splits X_all by the categorical inserted, afterwords it splits every column to two groups by the binary splitter.
performs a T-test between both groups and plots all result.
A chart for better explanation will be attached soon.
"""


def t_test_progress_over_categorical(X_all, categorical, splitter, categorical_name="", splitter_name=""):
    splitter_unique = splitter.unique()
    # Set some parameters for the plot
    fig, ax = plt.subplots()
    plot_len = len(categorical.unique())
    ind = np.arange(plot_len)
    ax.set_xticks(ind)
    ax.set_xticklabels(sorted(categorical.unique()))
    # The width of each bar and also the space between them.
    width = 0.15
    total_width = 0  # Total width will used as the locaton of the current bar.
    p_values_of_specific_column = []  # The progress of the P values of each column

    for col in X_all.columns:
        relevant_col = X_all[col]
        for unique_value in sorted(categorical.unique()):
            values_fulfill_condition = relevant_col[
                categorical == unique_value]  # split the column by the categoricl value.
            relevant_splitter = splitter[
                categorical == unique_value]  # Select the relevant spliter for the values above.
            p_val = stats.ttest_ind(values_fulfill_condition[relevant_splitter == splitter_unique[0]],
                                    values_fulfill_condition[relevant_splitter == splitter_unique[1]], equal_var=False)[
                1]
            p_values_of_specific_column.append(-1 * log10(p_val))
        ax.bar(ind + total_width, p_values_of_specific_column, width,
               label=col)  # Create bars describing the development of the P values.
        total_width += width  # update the location for the next bar group.
        p_values_of_specific_column.clear()

    # Creating the plot and saving it.
    ax.set_xlabel(categorical_name)
    ax.set_ylabel('-log(P values)')
    ax.set_title('T test progress over : ' + categorical_name)
    ax.axhline(y=-log10(Constants.P_VALUE_THRESHOLD), linewidth=1, color='r', ls='--', label='P = -log(0.05)')
    ax.legend(title="Splitted by: " + splitter_name)
    fig.savefig("T_test_" + categorical_name + "_" + splitter_name + ".png")
    plt.close()
