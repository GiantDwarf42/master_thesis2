#%%
print("this was modified by the server")
#%%
import numpy as np
#%%
import torch
#%%
torch.empty(3,4)
#%%
ar1 = np.arange(10)
ar2 = np.arange(10,20)
#%%
t1 = torch.tensor((ar1, ar2))
#%%
tr = torch.rand(2,10)
#%%
t1 * tr
#%%
trt = torch.transpose(tr,0,1)
torch.matmul(t1.float(), trt)
#%%
t1

#%%
trt.shape

#%%