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
load_path = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Logistic_Data_processed\Logistic_Processed_data_b0.01.pkl"
#load_path = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Logistic_Data_processed\Logistic_Processed_data_b0.1.pkl"
#load_path = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Logistic_Data_processed\Logistic_Processed_data_bAUTO.pkl"


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
df_last_150_proper_aggregation_index_reset.columns = index
df_last_150_proper_aggregation_index_reset

# %%
# %%
saving_path_result_data = f"/home/users/k/kipfer2/Logistics_processed_results/processed_tables"

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="b0.1"]
data_output.to_pickle(f"{saving_path_result_data}/Logistics_result_b0.1.pkl")

# %%
data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="b0.01"]
data_output.to_pickle(f"{saving_path_result_data}/Logistics_result_b0.01.pkl")

data_output = df_last_150_proper_aggregation_index_reset[df_last_150_proper_aggregation_index_reset["b"]=="bAUTO"]
data_output.to_pickle(f"{saving_path_result_data}/Logistics_result_bAUTO.pkl")




number_iterations = 2500
df = pd.concat([df, pd.DataFrame.from_dict({"Iteration": np.tile(np.arange(number_iterations),int(df.shape[0]/number_iterations))})], axis=1)
df

param_combinations = df.groupby(["alpha", "dim", "b", "xsize", "ysize"]).groups.keys()
param_combinations


saving_path_convergence = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Logistic_Convergence_charts"



