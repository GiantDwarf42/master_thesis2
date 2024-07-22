# %% [markdown]
# # Gaussian Data Analysis
# 
# This notebook is analyzing the Gaussian simulation study

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
number_iterations = 1500

load_path = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Gaussian_Data_processed\Gauss_Processed_data.pkl"

#loading the data
# df = pd.read_csv(load_path) # takes way longer and loads from csv source
df = pd.read_pickle(load_path)

df = df.drop(["time", "MMD"], axis=1)

# %%
df.shape[0] # the expected number of rows is 27000*1500 = 40500000, the simulation worked now


# %%
df.info()

# %% [markdown]
# # Notes
# 
# * For big sigma convergence is might not be achieved in 1500 iterations for all b. When b is selected intelligently or sigma small no issue.
# * b is important
# * bigger sigma needs more iteration => clear but important result
# * sample size doesn't seem to be very important

# %%


# %% [markdown]
# 

# %% [markdown]
# # Full Aggregation

# %%
def last_150_rows(group):
    return group.tail(150)

full_group_by = ["mu", "sigma", "b", "xsize", "ysize", "ID"]

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
df_last_150_proper_aggregation = df_last_150_sanity.reset_index().drop(["ID"], axis=1).groupby(["mu", "sigma", "b", "xsize", "ysize"]).agg(["count","mean", "std"])
df_last_150_proper_aggregation

# %%
df_last_150_proper_aggregation.columns = df_last_150_proper_aggregation.columns.droplevel(level=1)
df_last_150_proper_aggregation

# %%
df_last_150_proper_aggregation_index_reset = df_last_150_proper_aggregation.reset_index()
df_last_150_proper_aggregation_index_reset

# %%
df_last_150_proper_aggregation_index_reset.columns

# %%
index = pd.MultiIndex.from_tuples([
    ('mu', ''), ('sigma', ''), ('b', ''), ('simulation sample size', ''), ('target sample size', ''),
    ('mu_hat', 'count'), ('mu_hat', 'mean'), ('mu_hat', 'std'),
    ('sigma_hat', 'count'), ('sigma_hat', 'mean'), ('sigma_hat', 'std'),
    ('mu_hat_MLE', 'count'), ('mu_hat_MLE', 'mean'), ('mu_hat_MLE', 'std'),
    ('sigma_hat_MLE', 'count'), ('sigma_hat_MLE', 'mean'), ('sigma_hat_MLE', 'std')
])

# %%
df_last_150_proper_aggregation_index_reset.columns = index
df_last_150_proper_aggregation_index_reset

# %% [markdown]
# # Tables aggregated over the 150 simulations split by b values

# %%
saving_path_result_data = f"sdjfhj2ehgf"

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="b0.1"].set_index(["mu", "sigma", "b", 'simulation sample size', 'target sample size'])
data_output.to_csv(f"{saving_path_result_data}/Gaussian_result_b0.1.csv")

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="b0.01"].set_index(["mu", "sigma", "b", 'simulation sample size', 'target sample size'])
data_output.to_csv(f"{saving_path_result_data}/Gaussian_result_b0.01.csv")

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="bAUTO"].set_index(["mu", "sigma", "b", 'simulation sample size', 'target sample size'])
data_output.to_csv(f"{saving_path_result_data}/Gaussian_result_bAUTO.csv")

# %%


# %% [markdown]
# # Convergence plots

# %%
df = df.reset_index(drop=True)
df

# %%
df.columns

# %%
df = pd.concat([df, pd.DataFrame.from_dict({"Iteration": np.tile(np.arange(number_iterations),int(df.shape[0]/number_iterations))})], axis=1)
df

# %%
param_combinations = df.groupby(["mu", "sigma", "b", "xsize", "ysize"]).groups.keys()
param_combinations

# %%
saving_path_convergence = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Gaussian_Convergence_charts"

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

counter = 0
# Assuming param_combinations is already defined and df is your DataFrame
for mu, sigma, b, xsize, ysize in param_combinations:
    simulations_df = df[(df["mu"] == mu) & (df["sigma"] == sigma) 
                        & (df["b"] == b) & (df["xsize"] == xsize) 
                        & (df["ysize"] == ysize)]
    
    melted_df = simulations_df.melt(id_vars=["ID", "Iteration"],
                                    value_vars=["mu_hat", "sigma_hat"],
                                    var_name="param_type",
                                    value_name="param_value")

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Plot all lines for each ID
    for param_type, color in zip(["mu_hat", "sigma_hat"], ["blue", "orange"]):
        subset = melted_df[melted_df["param_type"] == param_type]
        sns.lineplot(data=subset, x='Iteration', y='param_value', units='ID', estimator=None, color=color, alpha=0.05)
    
    # Add horizontal reference lines for mu and sigma
    plt.axhline(y=mu, color="blue", linestyle='-', label=r'$\hat\mu$')
    plt.axhline(y=sigma, color="orange", linestyle='-', label=r'$\hat\sigma$')
    
    plt.ylabel("Parameter Value")
    plt.title(f"Convergence plot with $\mu$ = {mu}, $\sigma$ = {sigma} and {b}")

    # Adjust the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Parameter')
    
    file_name = f"mu{mu}_sigma{sigma}_b{b}_xsize{xsize}_ysize{ysize}.png"
    plt.savefig(f"{saving_path_convergence}/{file_name}")
    plt.close()

    counter += 1
    if counter == 1:
	
    	break


# %%


# %% [markdown]
# # Violin chart for sample sizes

# %%
saving_path_sample_size_effect = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Gaussian_Violin_sample_size_effect"

# %%
# getting the individual parameter combination sub dataframes 

df_last_150_aggregated_index_reset = df_last_150_aggregated.reset_index()


# Get the unique combinations of the multi-index levels

param_combinations = df_last_150_aggregated_index_reset.groupby(["mu", "sigma", "b"]).groups.keys()
param_combinations

# %% [markdown]
# ### Y sample size effect

# %%
# go through the different simulation runs => 180
counter = 0
# Iterate through each combination of index levels
for mu, sigma, b in param_combinations:

	simulations_df = df_last_150_aggregated_index_reset[(df_last_150_aggregated_index_reset["mu"]==mu) 
						     & (df_last_150_aggregated_index_reset["sigma"]==sigma) 
						     & (df_last_150_aggregated_index_reset["b"]==b)]
	
	# Reset the multi-index and flatten the columns
	simulations_df = simulations_df.reset_index()
	simulations_df.columns = ['_'.join(filter(None, col)).strip() for col in simulations_df.columns.values]

	# Rename the columns for easier access
	simulations_df.rename(columns={
	'ysize_': 'ysize',
	'xsize_': 'xsize',
	'mu_hat_mean': 'mu_hat_mean',
	'sigma_hat_mean': 'sigma_hat_mean'
	}, inplace=True)

	# Melt the DataFrame to long format for easier plotting with seaborn
	df_long = pd.melt(simulations_df, id_vars=['ysize', 'xsize'], value_vars=['mu_hat_mean', 'sigma_hat_mean'], 
			var_name='Type', value_name='Mean')

	# Create a FacetGrid for separate violin plots by xsize
	g = sns.FacetGrid(df_long, col="xsize", col_wrap=3, sharey=True, height=5, aspect=1)

	# Map the violinplot to the FacetGrid
	g.map_dataframe(sns.violinplot, x='ysize', y='Mean', hue='Type', split=True, inner='quartile', palette='muted', alpha=0.5)

	# Add horizontal lines to each subplot
	for ax in g.axes.flat:
		ax.axhline(y=simulations_df["mu"].iloc[0], color='blue', linestyle='--', linewidth=1)
		ax.axhline(y=simulations_df["sigma"].iloc[0], color='orange', linestyle='-', linewidth=1)

	# Get handles and labels from the first axis
	handles, labels = g.axes.flat[0].get_legend_handles_labels()

	# Replace old labels with LaTeX formatted ones

	
	new_labels = [r'$\hat{\mu}$' if label == 'mu_hat_mean' else r'$\hat{\sigma}$' for label in labels]

	# Manually add the new legend
	# Create a new legend with the handles and updated labels
	for ax in g.axes.flat:
		ax.legend(handles=handles, labels=new_labels, title='Parameter')

	# Set titles and axis labels
	g.set_axis_labels('response sample size', 'Parameter Values')
	g.set_titles(col_template='simulations distribution sample size : {col_name}')

	# Adjust the main title
	plt.subplots_adjust(top=0.93)

	mu_latex = r"$\mu$"
	sigma_latex = r"$\sigma$"
	g.figure.suptitle(f'Effect of the response sample size with  {mu_latex} = {mu}, {sigma_latex} = {sigma} and {b}')

	file_name = f"mu{mu}_sigma{sigma}_{b}_yeffect"

	#plt.savefig(f"{saving_path_sample_size_effect}\{file_name}.png")

	plt.show()
	plt.close()

	counter += 1

	print(f"created and saved {counter} figures")

	if counter == 1:
		break
	

	

# %% [markdown]
# ### xsize Effect

# %%
# go through the different simulation runs => 180
counter = 0
# Iterate through each combination of index levels
for mu, sigma, b in param_combinations:

	simulations_df = df_last_150_aggregated_index_reset[(df_last_150_aggregated_index_reset["mu"]==mu) 
						     & (df_last_150_aggregated_index_reset["sigma"]==sigma) 
						     & (df_last_150_aggregated_index_reset["b"]==b)]
	
	# Reset the multi-index and flatten the columns
	simulations_df = simulations_df.reset_index()
	simulations_df.columns = ['_'.join(filter(None, col)).strip() for col in simulations_df.columns.values]

	# Rename the columns for easier access
	simulations_df.rename(columns={
	'ysize_': 'ysize',
	'xsize_': 'xsize',
	'mu_hat_mean': 'mu_hat_mean',
	'sigma_hat_mean': 'sigma_hat_mean'
	}, inplace=True)

	# Melt the DataFrame to long format for easier plotting with seaborn
	df_long = pd.melt(simulations_df, id_vars=['ysize', 'xsize'], value_vars=['mu_hat_mean', 'sigma_hat_mean'], 
			var_name='Type', value_name='Mean')

	# Create a FacetGrid for separate violin plots by xsize
	g = sns.FacetGrid(df_long, col="ysize", col_wrap=3, sharey=True, height=5, aspect=1)

	# Map the violinplot to the FacetGrid
	g.map_dataframe(sns.violinplot, x='xsize', y='Mean', hue='Type', split=True, inner='quartile', palette='muted', alpha=0.5)

	# Add horizontal lines to each subplot
	for ax in g.axes.flat:
		ax.axhline(y=simulations_df["mu"].iloc[0], color='blue', linestyle='--', linewidth=1)
		ax.axhline(y=simulations_df["sigma"].iloc[0], color='orange', linestyle='-', linewidth=1)

	# Get handles and labels from the first axis
	handles, labels = g.axes.flat[0].get_legend_handles_labels()

	# Replace old labels with LaTeX formatted ones

	
	new_labels = [r'$\hat{\mu}$' if label == 'mu_hat_mean' else r'$\hat{\sigma}$' for label in labels]

	# Manually add the new legend
	# Create a new legend with the handles and updated labels
	for ax in g.axes.flat:
		ax.legend(handles=handles, labels=new_labels, title='Parameter')

	# Set titles and axis labels
	g.set_axis_labels('simulation sample size', 'Parameter Values')
	g.set_titles(col_template='response sample size: {col_name}')

	# Adjust the main title
	plt.subplots_adjust(top=0.93)

	mu_latex = r"$\mu$"
	sigma_latex = r"$\sigma$"
	g.figure.suptitle(f'Effect of the simulation sample size with  {mu_latex} = {mu}, {sigma_latex} = {sigma} and {b}')

	file_name = f"mu{mu}_sigma{sigma}_{b}_xeffect"

	#plt.savefig(f"{saving_path_sample_size_effect}\{file_name}.png")

	plt.show()
	plt.close()

	counter += 1

	print(f"created and saved {counter} figures")

	if counter == 1:
		break
	

	

# %%



