# %% [markdown]
# # Gaussian Sim study analysis

# %%
import os
import pandas as pd
import numpy as np
from collections import defaultdict


# %%

# Step 1: Specify the directory containing the CSV files
directory = r'G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Gaussian_Data_raw'


# %%

# Step 2: List all files in the directory
all_files = os.listdir(directory)


# %%

# Step 3: Group files by their common components except the ID
file_groups = defaultdict(list)
for file in all_files:
    if file.endswith('.csv'):
        parts = file.split('_')
        key = '_'.join(parts[:-1])  # Everything except the last part (ID)
        file_groups[key].append(file)


# %%

# Step 4: Read and stack files within each group, adding columns for the components
stacked_dataframes = []
count = 0
for key, files in file_groups.items():
    dataframes = []
    for file in sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0][2:])):
        df = pd.read_csv(os.path.join(directory, file),index_col=False)
    
        df = df.drop(["Unnamed: 0", "b"], axis=1)
        
        # Extract components from the filename
        parts = file.split('_')

        
        components = {
            'mu': np.repeat(int(parts[0][2:]), df.shape[0]),
            'sigma':np.repeat( int(parts[1][5:]),df.shape[0]),
            'xsize': np.repeat(int(parts[2][5:]),df.shape[0]),
            'ysize': np.repeat(int(parts[3][5:]),df.shape[0]),
            'b': np.repeat(parts[4],df.shape[0]),
            'ID': np.repeat(int(parts[5].split('.')[0][2:]),df.shape[0])  # Remove the .csv part
        }

        flag_df = pd.DataFrame.from_dict(components)

        df_flags_attached = pd.concat([df,flag_df], axis=1)
        
        dataframes.append(df_flags_attached)

        break

    
    # Concatenate dataframes
    stacked_df = pd.concat(dataframes, ignore_index=True)
    stacked_dataframes.append(stacked_df)
    
    count += 1

    print(f"loaded and processed {count} simulation")
    


# Now `stacked_dataframes` is a list of stacked dataframes with additional columns for the components.

# stack everything => same name for ram issue...
stacked_dataframes = pd.concat(stacked_dataframes)  


# %%
stacked_dataframes.info()

# %%
# 5 step: safe the processed DataFrame
file_name = "Gauss_Processed_data"
saving_path = r"G:\My Drive\Studium\UNIGE_Master\Thesis\Master_Thesis\Data\Gaussian_Data_processed"

stacked_dataframes.to_csv(f"{saving_path}\{file_name}.csv")

# %%
stacked_dataframes.to_pickle(f"{saving_path}\{file_name}.pkl")

# %%


# %%


# %%


# %%


# %%


# %%



