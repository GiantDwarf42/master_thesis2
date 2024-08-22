# %% [markdown]
# # Hüsler Reiss Data Analysis
# 
# This notebook is analyzing the Hüssler Reiss simulation study

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = 500



# %%
load_path = r"/home/users/k/kipfer2/Huesler_Reiss_processed_results_p/Data/Huesler_Reiss_Processed_data_p.pkl"

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
# # Full Aggregation

# %%
# %%
def last_150_rows(group):
    return group.tail(150)

full_group_by = ["grid", "alpha", "p", "xsize", "ysize", "ID"]

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
df_last_150_proper_aggregation = df_last_150_sanity.reset_index().drop(["ID"], axis=1).groupby(["grid", "alpha", "p", "xsize", "ysize"]).agg(["count","mean", "std"])
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
saving_path_result_data = r"/home/users/k/kipfer2/Huesler_Reiss_processed_results_p/processed_tables"

# %%

data_output = df_last_150_proper_aggregation_index_reset
data_output.to_pickle(f"{saving_path_result_data}/Huesler_Reiss_result_p.pkl")
data_output.to_csv(f"{saving_path_result_data}/Huesler_Reiss_result_p.csv")

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
param_combinations = df.groupby(["grid", "alpha", "p", "xsize", "ysize"]).groups.keys()
param_combinations

# %%
saving_path_convergence = r"/home/users/k/kipfer2/Huesler_Reiss_processed_results_p/convergence_charts"

# %% [markdown]
# # Convergence Charts

counter = 0
# Assuming param_combinations is already defined and df is your DataFrame
for grid, alpha, p, xsize, ysize in param_combinations:
    simulations_df = df[(df["grid"] == grid) & (df["alpha"] == alpha) 
                        & (df["p"] == p) & (df["xsize"] == xsize) 
                        & (df["ysize"] == ysize)]
    
    melted_df = simulations_df.melt(id_vars=["ID", "Iteration"],
                                    value_vars=["alpha_hat", "p_hat"],
                                    var_name="param_type",
                                    value_name="param_value")

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Plot all lines for each ID
    for param_type, color in zip(["alpha_hat", "p_hat"], ["blue", "orange"]):
        subset = melted_df[melted_df["param_type"] == param_type]
        sns.lineplot(data=subset, x='Iteration', y='param_value', units='ID', estimator=None, color=color, alpha=0.5)
    
    # Add horizontal reference lines for mu and sigma
    plt.axhline(y=alpha, color="blue", linestyle='-', label=r'$\hat\alpha$')
    plt.axhline(y=p, color="orange", linestyle='-', label=r'$\hat{p}$')
    
    plt.ylabel("Parameter Value")
    alpha_latex = r"$\alpha$"
    grid_name = f"{grid}"
    plt.title(f"Convergence plot with {alpha_latex} = {alpha}, $p$ = {p} and grid = {grid_name}")

    # Adjust the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Parameter')
    
    file_name = f"grid{grid}_alpha{alpha}_p{p}_xsize{xsize}_ysize{ysize}_300min.png"
    plt.savefig(f"{saving_path_convergence}/{file_name}")
    plt.close()

    counter += 1

    print(f"created and saved {counter} figures")
   
#%%

# %%


# %% [markdown]
# # Violin Charts

# %%
saving_path_sample_size_effect = r"/home/users/k/kipfer2/Huesler_Reiss_processed_results_p/violin_charts"

# %%
df_last_150_aggregated_index_reset = df_last_150_aggregated.reset_index()

# %%
param_combinations = df_last_150_aggregated_index_reset.groupby(["grid", "alpha", "p"]).groups.keys()
param_combinations

# %%
df_last_150_aggregated_index_reset


# %%
# ### Y sample size effect

#%%

# go through the different simulation runs => 180
counter = 0
# Iterate through each combination of index levels
for grid, alpha, p in param_combinations:

	simulations_df = df_last_150_aggregated_index_reset[(df_last_150_aggregated_index_reset["grid"]==grid) 
						     & (df_last_150_aggregated_index_reset["p"]==p) 
						     & (df_last_150_aggregated_index_reset["alpha"]==alpha)]
	
	# Reset the multi-index and flatten the columns
	simulations_df = simulations_df.reset_index()
	simulations_df.columns = ['_'.join(filter(None, col)).strip() for col in simulations_df.columns.values]

	
	# Rename the columns for easier access
	
	# simulations_df.rename(columns={
	# 'ysize_': 'ysize',
	# 'xsize_': 'xsize',
	# 'alpha_hat_mean': 'alpha_hat_mean',
	# 'p_hat_mean': 'p_hat_mean'
	# }, inplace=True)

	

	# Melt the DataFrame to long format for easier plotting with seaborn
	df_long = pd.melt(simulations_df, id_vars=['ysize', 'xsize'], value_vars=['alpha_hat_mean', 'p_hat_mean'], 
			var_name='Type', value_name='Mean')

	# Create a FacetGrid for separate violin plots by xsize
	g = sns.FacetGrid(df_long, col="xsize", col_wrap=3, sharey=True, height=5, aspect=1)

	# Map the violinplot to the FacetGrid
	g.map_dataframe(sns.violinplot, x='ysize', y='Mean', hue='Type', split=True, inner='quartile', palette='muted', alpha=0.5)

	# Add horizontal lines to each subplot
	for ax in g.axes.flat:
		ax.axhline(y=simulations_df["alpha"].iloc[0], color='blue', linestyle='--', linewidth=1)
		ax.axhline(y=simulations_df["p"].iloc[0], color='orange', linestyle='-', linewidth=1)

	# Get handles and labels from the first axis
	handles, labels = g.axes.flat[0].get_legend_handles_labels()

	# Replace old labels with LaTeX formatted ones

	
	new_labels = [r'$\hat{\alpha}$' if label == 'alpha_hat_mean' else r'$\hat{p}$' for label in labels]

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
	grid_name = f"{grid}"
	g.figure.suptitle(f'Effect of the response sample size with  {alpha_latex} = {alpha}, p = {p} and grid = {grid_name}')

	file_name = f"alpha{alpha}_p{p}_grid{grid}_yeffect_300min"

	plt.savefig(f"{saving_path_sample_size_effect}/{file_name}.png")

	plt.close()

	counter += 1


	print(f"created and saved {counter} figures")





#######################################################################

# %% [markdown]
# # xsample size effect

# %%

# go through the different simulation runs => 180
counter = 0
# Iterate through each combination of index levels
for grid, alpha, p in param_combinations:

	simulations_df = df_last_150_aggregated_index_reset[(df_last_150_aggregated_index_reset["grid"]==grid) 
						     & (df_last_150_aggregated_index_reset["alpha"]==alpha) 
						     & (df_last_150_aggregated_index_reset["p"]==p)]
	
	# Reset the multi-index and flatten the columns
	simulations_df = simulations_df.reset_index()
	simulations_df.columns = ['_'.join(filter(None, col)).strip() for col in simulations_df.columns.values]

	# # Rename the columns for easier access
	# simulations_df.rename(columns={
	# 'ysize_': 'ysize',
	# 'xsize_': 'xsize',
	# 'mu_hat_mean': 'mu_hat_mean',
	# 'sigma_hat_mean': 'sigma_hat_mean'
	# }, inplace=True)

	# Melt the DataFrame to long format for easier plotting with seaborn
	df_long = pd.melt(simulations_df, id_vars=['ysize', 'xsize'], value_vars=['alpha_hat_mean', 'p_hat_mean'], 
			var_name='Type', value_name='Mean')

	# Create a FacetGrid for separate violin plots by xsize
	g = sns.FacetGrid(df_long, col="ysize", col_wrap=3, sharey=True, height=5, aspect=1)

	# Map the violinplot to the FacetGrid
	g.map_dataframe(sns.violinplot, x='xsize', y='Mean', hue='Type', split=True, inner='quartile', palette='muted', alpha=0.5)

	# Add horizontal lines to each subplot
	for ax in g.axes.flat:
		ax.axhline(y=simulations_df["alpha"].iloc[0], color='blue', linestyle='--', linewidth=1)
		ax.axhline(y=simulations_df["p"].iloc[0], color='orange', linestyle='-', linewidth=1)

	# Get handles and labels from the first axis
	handles, labels = g.axes.flat[0].get_legend_handles_labels()

	# Replace old labels with LaTeX formatted ones

	
	new_labels = [r'$\hat{\alpha}$' if label == 'alpha_hat_mean' else r'$\hat{p}$' for label in labels]

	# Manually add the new legend
	# Create a new legend with the handles and updated labels
	for ax in g.axes.flat:
		ax.legend(handles=handles, labels=new_labels, title='Parameter')

	# Set titles and axis labels
	g.set_axis_labels('simulation sample size', 'Parameter Values')
	g.set_titles(col_template='response sample size: {col_name}')

	# Adjust the main title
	plt.subplots_adjust(top=0.93)

	alpha_latex = r"$\alpha$"
	grid_name = f"{grid}"
	g.figure.suptitle(f'Effect of the simulation sample size with  {alpha_latex} = {alpha}, p = {p} and grid = {grid_name}')

	file_name = f"alpha{alpha}_p{p}_grid{grid}_xeffect_300min"

	plt.savefig(f"{saving_path_sample_size_effect}/{file_name}.png")

	plt.close()

	counter += 1

	print(f"created and saved {counter} figures")


	

# %%


# %%


# %%


# %%


# %%



