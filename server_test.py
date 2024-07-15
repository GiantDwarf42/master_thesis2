#%%
import torch
cuda_available = torch.cuda.is_available()
#%%
import datetime
#%%
import numpy as np
#%%
ar1 = np.arange(10)
ar2 = np.arange(10,20)
#%%
# set the device for computation on cpu or gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Cuda available?", torch.cuda.is_available(), "device: ",device)
#%%
t1 = torch.tensor((ar1, ar2))
t1 = t1.to(device)
#%%
tr = torch.rand(2,10, device=device)
#%%
component_product = t1 * tr
component_sum = t1 + tr
#%%
trt = torch.transpose(tr,0,1)
matrix_product = torch.matmul(t1.float(), trt)
now = datetime.datetime.now()

#%%
with open("test_output.txt", "w") as text_file:
	text_file.write(f"Execution worked at time {now.time()}\n")
	text_file.write(f"Cuda available: {cuda_available}, device: {device}\n")
	text_file.write(f"Tensor1: {t1}\n")
	text_file.write(f"Tensor Random: {tr}\n")
	text_file.write(f"Component sum: {component_sum}\n")
	text_file.write(f"Component product: {component_product}\n")
	text_file.write(f"Matrix product: {matrix_product}\n")
	text_file.write(f"file overwritten\n")
#%%

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%