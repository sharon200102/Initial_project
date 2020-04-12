import itertools
import seaborn as sns
import matplotlib.pyplot as plt

"""
             Iterate through all n chose two sub groups of columns in the data, and plot the relationship between them.
             colored by the categorical columns inserted.


"""


def visualize_in_pairs(X_all, data_name="", categorical_columns=None, figsize=(18, 18), **kwargs):
    """If there is no categorical columns to color by """
    if len(categorical_columns) == 0:
        pairplot = sns.pairplot(X_all, **kwargs)
        pairplot.savefig(data_name + '_pairplot_without_coloring.png')
        plt.close()


    else:
        # Iterate through all categorical columns
        for (categoricalName, categoricalData) in categorical_columns.iteritems():
            # For each one create a triangular subplots grid.
            fig, axes = plt.subplots(X_all.shape[1], X_all.shape[1], squeeze=False, figsize=figsize)
            # Iterate through all subgroups
            for row, col in list(itertools.combinations(list(range(0, X_all.shape[1])), 2)):
                # In case that the relevant categorical column is not in Integer type transform it to be.
                try:
                    axes[row][col].scatter(X_all.iloc[:, row], X_all.iloc[:, col], c=categoricalData, **kwargs)


                except ValueError:
                    categoricalData = categoricalData.transform(lambda x: list(categoricalData.unique()).index(x))
                    axes[row][col].scatter(X_all.iloc[:, row], X_all.iloc[:, col], c=categoricalData, **kwargs)

                finally:
                    # Shape the subplot to be in a  triangular form and set titles.
                    fig.delaxes(axes[col][row])
                    axes[row][col].set_xlabel(X_all.columns[row])
                    axes[row][col].set_ylabel(X_all.columns[col])
            for diag in range(0, X_all.shape[1]):
                fig.delaxes(axes[diag][diag])

            # Set a main title and save the figure.
            fig.suptitle(data_name + '_pairplot_' + categoricalName + '_coloring')
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            fig.savefig(data_name + '_pairplot_' + categoricalName + '_coloring.png')
            plt.close()

"""
At the moment: the function plots the progress of the mean of each column,
based on the unique values of the categorical column inserted. 
"""
def column_attribute_progress_in_categorical(X_all, categorical_series, categorical_name="categorical",splitter=None,splitter_name=""):
    # Marker_counter is equal to 2 because the previous markers in the list are difficult to see.
    # the marker will be responsible to the difference between the columns.

    marker_counter = 0
    fig = plt.figure(figsize=(10,10))
    new_plot = fig.add_subplot(111)
    # if there is no splitter
    if len(splitter)==0:

        # Add the categorical column to the data.
        X_all = X_all.assign(categorical=categorical_series.values)
        # Group the data based on the categorical column.
        grouped = X_all.groupby(['categorical']).mean()
        # Every column in grouped contains the mean of the column for every unique value in categorical_series
        for col_name in grouped.columns:
            # Plot the line the goes through the means of the column in the different categorical values.
            new_plot.plot(grouped.index, grouped[col_name], marker=marker_counter, markersize=10,
                          label=col_name)
            marker_counter += 1

        #Plot A vertical line for each unique categorical value.

        for time_point in grouped.index:
            ymin = grouped.loc[time_point].min()
            ymax = grouped.loc[time_point].max()
            new_plot.plot([time_point,time_point],[ymin,ymax],c='k')
    # Splitter exists
    else:
        # Add both of the categorical columns
        X_all = X_all.assign(categorical=categorical_series.values,splitter=splitter)
        # Groupby noth columns and make only categorical as index avoiding two indices situation.
        grouped = X_all.groupby(['categorical','splitter']).mean().reset_index(level=1)
        # save and remove the splitter because its not truly a part of the data,it was only added because of the groupby
        splitter_index=grouped['splitter']
        grouped.drop('splitter',axis=1,inplace=True)

#   The columns in grouped represent the progress of the original column over categorical.

        for col_name in grouped.columns:
            #Select every column by all possible splitter values and plot the progress.
            for splitter_val in splitter_index.unique():
                splitter_progress=grouped[col_name][splitter_index==splitter_val]
                new_plot.plot(splitter_progress.index, splitter_progress, marker=marker_counter, markersize=10,
                              label=col_name+"_"+str(splitter_val))
                marker_counter += 1

        #Plot A vertical line for each unique categorical value.
        for time_point in grouped.index.unique():
            grouped_in_specific_time_point=grouped[grouped.index==time_point]
            ymin =max(grouped_in_specific_time_point.max(axis=1))
            ymax = min(grouped_in_specific_time_point.min(axis=1))
            new_plot.plot([time_point,time_point],[ymin,ymax],c='k')



    new_plot.set_xlabel(categorical_name)
    new_plot.set_ylabel('Mean value')
    new_plot.set_title('Columns progress in ' + categorical_name)
    plt.legend(title="Splitted by: "+splitter_name)
    plt.tight_layout()
    fig.savefig('Columns_progress_in_' + categorical_name +"_"+splitter_name+'.png')
    plt.close()








