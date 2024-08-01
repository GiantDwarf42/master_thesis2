# %% [markdown]
# # Logistic Data Analysis
# 
# This notebook is analyzing the Logistic simulation study

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = 500

# %% [markdown]
# # b0.01 case
# 

# %%
load_path = r"/home/users/k/kipfer2/Logistic_processed_results/Data/Logistic_Processed_data.pkl"

#loading the data

df = pd.read_pickle(load_path)

df

# %%
df.shape[0] #


# %%
df.info()

# %% [markdown]
# # Notes
# 
# 

# %% [markdown]
# * A LOT of NaN values
# * when b0.01 or 0.1 it's even worse, AUTO is slightly better
# * 0.05, 0.1 and 0.95 fail, 0.3 and 0.7 seem to work more or less
# * estimation gets better with bigger d, higher xsize and higher ysize

# %% [markdown]
# # Full Aggregation

# %%
df

# %%
# %%
def last_150_rows(group):
    return group.tail(150)

full_group_by = ["alpha", "dim", "b", "xsize", "ysize", "ID"]

df_last_150 = df.groupby(full_group_by).apply(last_150_rows).reset_index(drop=True)

df_last_150

# %%
df_last_150_aggregated = df_last_150.groupby(full_group_by).agg(["count", 'mean', "std"])
df_last_150_aggregated

# %% [markdown]
# # correct aggregation
# 

# %%
df_last_150_sanity = df_last_150.groupby(full_group_by).agg(['mean'])
df_last_150_sanity_index_reset = df_last_150_sanity.reset_index()
df_last_150_sanity_index_reset

# %%
df_last_150_proper_aggregation = df_last_150_sanity.reset_index().drop(["ID"], axis=1).groupby(["alpha", "dim", "b", "xsize", "ysize"]).agg(["count","mean", "std"])
df_last_150_proper_aggregation

# %%
df_last_150_proper_aggregation.columns = df_last_150_proper_aggregation.columns.droplevel(level=1)
df_last_150_proper_aggregation

# %%
df_last_150_proper_aggregation_index_reset = df_last_150_proper_aggregation.reset_index()
df_last_150_proper_aggregation_index_reset.head()

# %%


# %%




# %%
# %%
saving_path_result_data = f"/home/users/k/kipfer2/Logistic_processed_results/processed_tables"

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="b0.1"]
data_output.to_pickle(f"{saving_path_result_data}/Logistics_result_b0.1.pkl")
data_output.to_csv(f"{saving_path_result_data}/Logistics_result_b0.1.csv")

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="b0.01"]
data_output.to_pickle(f"{saving_path_result_data}/Logistics_result_b0.01.pkl")
data_output.to_csv(f"{saving_path_result_data}/Logistics_result_b0.01.csv")

data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="bAUTO"]
data_output.to_pickle(f"{saving_path_result_data}/Logistics_result_bAUTO.pkl")
data_output.to_csv(f"{saving_path_result_data}/Logistics_result_bAUTO.csv")

# %% [markdown]
# # Convergence plots b0.01 case

# %%
df = df.reset_index(drop=True)
df

# %%
df.columns


# %%
number_iterations = 2500
df = pd.concat([df, pd.DataFrame.from_dict({"Iteration": np.tile(np.arange(number_iterations),int(df.shape[0]/number_iterations))})], axis=1)
df

# %%
param_combinations = df.groupby(["alpha", "dim", "b", "xsize", "ysize"]).groups.keys()
param_combinations

# %%
saving_path_convergence = r"/home/users/k/kipfer2/Logistic_processed_results/convergence_charts"

# %% [markdown]
# # Convergence Charts

# %%
counter = 0
for alpha, dim, b, xsize, ysize in param_combinations:
    simulations_df = df[(df["alpha"]==alpha) & (df["dim"]==dim) 
		     & (df["b"]==b) & (df["xsize"]==xsize) & (df["ysize"]==ysize)]
	
    melted_df = simulations_df.melt(id_vars=["ID", "Iteration"],
		    value_vars=["alpha_hat"],
		    var_name = "param_type",
		    value_name = "param_value")
    

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Plot all lines for each ID
    for param_type, color in zip(["alpha_hat"], ["blue"]):
        subset = melted_df[melted_df["param_type"] == param_type]
        sns.lineplot(data=subset, x='Iteration', y='param_value', units='ID', estimator=None, color=color, alpha=0.1)
    
    # Add horizontal reference lines for mu and sigma
    plt.axhline(y=alpha, color="blue", linestyle='-', label=r'$\hat\alpha$')
    
    
    plt.ylabel("Parameter Value")

    alpha_symbol = r'$\alpha$'
    plt.title(f"Convergence plot with {alpha_symbol} = {alpha}, dimension {dim} and {b}")

    # Adjust the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Parameter')
    
    file_name = f"alpha{alpha}_dim{dim}_b{b}_xsize{xsize}_ysize{ysize}.png"
    plt.savefig(f"{saving_path_convergence}/{file_name}")
    #plt.show()
    plt.close()

    counter += 1
    print(f"created and saved {counter} figures")



# %%


# %% [markdown]
# # Violin Charts

# %%
saving_path_sample_size_effect = r"/home/users/k/kipfer2/Logistic_processed_results/violin_charts"

# %%
df_last_150_aggregated_index_reset = df_last_150_aggregated.reset_index()

# %%
param_combinations = df_last_150_aggregated_index_reset.groupby(["alpha", "dim", "b"]).groups.keys()
param_combinations

# %%
df_last_150_aggregated_index_reset

# %% [markdown]
# # ysample size
# 

# %%
# ### Y sample size effect

# %%
# go through the different simulation runs => 180
counter = 0
# Iterate through each combination of index levels
for alpha, dim, b in param_combinations:

	simulations_df = df_last_150_aggregated_index_reset[(df_last_150_aggregated_index_reset["alpha"]==alpha) 
						     & (df_last_150_aggregated_index_reset["dim"]==dim) 
						     & (df_last_150_aggregated_index_reset["b"]==b)]
	
	# Reset the multi-index and flatten the columns
	simulations_df = simulations_df.reset_index()
	simulations_df.columns = ['_'.join(filter(None, col)).strip() for col in simulations_df.columns.values]

	# # Melt the DataFrame to long format for easier plotting with seaborn
	df_long = pd.melt(simulations_df, id_vars=['ysize', 'xsize'], value_vars=['alpha_hat_mean'], 
			var_name='Type', value_name='Mean')

	# Create a FacetGrid for separate violin plots by xsize
	g = sns.FacetGrid(df_long, col="xsize", col_wrap=3, sharey=True, height=5, aspect=1)

	# Map the violinplot to the FacetGrid
	g.map_dataframe(sns.violinplot, x='ysize', y='Mean', hue='Type', split=True, inner='quartile', palette='muted', alpha=0.5)

	# Add horizontal lines to each subplot
	for ax in g.axes.flat:
		ax.axhline(y=alpha, color='blue', linestyle='--', linewidth=1)
		

	# Get handles and labels from the first axis
	handles, labels = g.axes.flat[0].get_legend_handles_labels()

	# Replace old labels with LaTeX formatted ones

	
	new_labels = [r'$\hat{\alpha}$'for label in labels]

	# Manually add the new legend
	# Create a new legend with the handles and updated labels
	for ax in g.axes.flat:
		ax.legend(handles=handles, labels=new_labels, title='Parameter')

	# Set titles and axis labels
	g.set_axis_labels('response sample size', 'Parameter Values')
	g.set_titles(col_template='simulations distribution sample size : {col_name}')

	# Adjust the main title
	plt.subplots_adjust(top=0.93)

	alpha_latex = r"$\alpha$"

	g.figure.suptitle(f'Effect of the response sample size with  {alpha_latex} = {alpha}, dimension = {dim} and {b}')

	file_name = f"alpha{alpha}_dim{dim}_{b}_yeffect"

	plt.savefig(f"{saving_path_sample_size_effect}/{file_name}.png")
	#plt.show()

	plt.close()

	counter += 1

	print(f"created and saved {counter} figures")



	

# %% [markdown]
# # xsample size effect

# %%


# %%
# ### Y sample size effect

# %%
# go through the different simulation runs => 180
counter = 0
# Iterate through each combination of index levels
for alpha, dim, b in param_combinations:

	simulations_df = df_last_150_aggregated_index_reset[(df_last_150_aggregated_index_reset["alpha"]==alpha) 
						     & (df_last_150_aggregated_index_reset["dim"]==dim) 
						     & (df_last_150_aggregated_index_reset["b"]==b)]
	
	# Reset the multi-index and flatten the columns
	simulations_df = simulations_df.reset_index()
	simulations_df.columns = ['_'.join(filter(None, col)).strip() for col in simulations_df.columns.values]

	# # Melt the DataFrame to long format for easier plotting with seaborn
	df_long = pd.melt(simulations_df, id_vars=['ysize', 'xsize'], value_vars=['alpha_hat_mean'], 
			var_name='Type', value_name='Mean')

	# Create a FacetGrid for separate violin plots by xsize
	g = sns.FacetGrid(df_long, col="ysize", col_wrap=3, sharey=True, height=5, aspect=1)

	# Map the violinplot to the FacetGrid
	g.map_dataframe(sns.violinplot, x='xsize', y='Mean', hue='Type', split=True, inner='quartile', palette='muted', alpha=0.5)

	# Add horizontal lines to each subplot
	for ax in g.axes.flat:
		ax.axhline(y=alpha, color='blue', linestyle='--', linewidth=1)
		

	# Get handles and labels from the first axis
	handles, labels = g.axes.flat[0].get_legend_handles_labels()

	# Replace old labels with LaTeX formatted ones

	
	new_labels = [r'$\hat{\alpha}$'for label in labels]

	# Manually add the new legend
	# Create a new legend with the handles and updated labels
	for ax in g.axes.flat:
		ax.legend(handles=handles, labels=new_labels, title='Parameter')

	# Set titles and axis labels
	g.set_axis_labels('response sample size', 'Parameter Values')
	g.set_titles(col_template='simulations distribution sample size : {col_name}')

	# Adjust the main title
	plt.subplots_adjust(top=0.93)

	alpha_latex = r"$\alpha$"

	g.figure.suptitle(f'Effect of the simulation sample size with  {alpha_latex} = {alpha}, dimension = {dim} and {b}')

	file_name = f"alpha{alpha}_dim{dim}_{b}_xeffect"

	plt.savefig(f"{saving_path_sample_size_effect}/{file_name}.png")
	#plt.show()

	plt.close()

	counter += 1

	print(f"created and saved {counter} figures")



	

# %%


# %%


# %%


# %%


# %%



