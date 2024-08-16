
import torch
import numpy as np
import pandas as pd
import time
from scipy.stats import genextreme
import matplotlib.pyplot as plt


def kernel_gauss_cdist(cdist:torch.tensor, b:int=1)->torch.tensor:
    """calculates the tensor (matrix) of gaussian kernel values calculated using cdist

    Args:
        cdist (torch.tensor): a 2d tensor of cdist output values
        b (int, optional): the bandwidth. Defaults to 1.

    Returns:
        torch.tensor: the resulting gaussian kernel value tensor 
    """

    res = torch.exp(-b*((cdist)**2))

    return res


def calc_b_heuristic(y:torch.tensor, n:int, case:str, device, params:dict)->torch.tensor:
    """calculates the median heuristic for b

    Args:
        y (torch.tensor): the target distribution tensor
        n (int): the number of rows the simulation distribution should
        case (str): the distribution for which to calculate b for cases: "norm", "logistic"
        device (_type_): the pytorch device object
        params (dict): a correctly named dictionary with the required parameters for the b calculation if "norm" => mu, sigma if "logistic" alpha
    Returns:
        torch.tensor: returns the bandwidth value as a 1x1 tensor
    """


    #case distinction
    if case == "norm":
        
        #this is a safety thing, during optimization it can incidentially happen
        # that sigma_hat becomes negative. It is very rare but it does.
        if params["sigma"] <= 0:
            params["sigma"] = 0.01
        
        x = torch.normal(params["mu"], params["sigma"],(n,y.shape[1]))
        b = 1/ torch.median(torch.cdist(x,y,p=2)**2)

    elif case == "logistic":
            
            #safety thing as above, might not be needed
            if params["alpha"] <= 0:
                params["alpha"] = 0.01
            elif params["alpha"] >= 1:
                params["alpha"] = 0.99

            x = sample_multivariate_logistic(n,y.shape[1], params["alpha"], device)
            b = 1/ torch.median(torch.cdist(x,y,p=2)**2)

    elif case == "huesler_reis":
        
        grid = params["grid"]

        Vario = params["Vario"]


        x = sim_huesler_reis_ext(grid, Vario, device, no_simu=n)
        
        b = 1/ torch.median(torch.cdist(x,y,p=2)**2)
         

    return b

def training_loop_gauss(mu_hat, sigma_hat,target_dist, nr_iterations , sample_size, device, b, optimizer, epoch_print_size=500, b_update=0)->pd.DataFrame:
    """workhorse function which does the actual gradient optimization in 
    the Gaussian case. This function is called for each simulation in a loop

    Args:
        mu_hat (_type_): The tensor for the mu parameter
        sigma_hat (_type_): the tensor for the sigma parameter
        target_dist (_type_): the target distribution vector
        nr_iterations (_type_): the nr of itterations per simulation (epochs)
        sample_size (_type_): the sample size for the simulation distribution
        device (_type_): the torch device object
        b (_type_): the bandwith parameter b
        optimizer (_type_): the setup optimizer object
        epoch_print_size (int, optional): the number of iterations after which a progress print should be made. Defaults to 500.
        b_update (int, optional) : the bandwith value will be adjusted every given number of iterations

    Returns:
        pd.DataFrame: the df with the results
    """
    mu_hat_estimates = []
    sigma_hat_estimates = []
    MMD_values = []
    b_values = []
    times = []

    start_time = time.time()

    #the MMD yy case is a constant.
    # no need to recalculate this for every epoch
    MMD_yy_case = MMD_equal_case(target_dist,device,b)
    
    for epoch in np.arange(nr_iterations):


        #iterative update of bandwidth value
        if b_update:

            if epoch % b_update == 0:

                params = {"mu": mu_hat.item(),
                          "sigma": sigma_hat.item()}
                
                
                b = calc_b_heuristic(target_dist, sample_size, "norm", device, params).item()

                MMD_yy_case = MMD_equal_case(target_dist,device,b)

        # Empty gradient
        optimizer.zero_grad()

        # Sample from the generator
        sample = torch.normal(0,1,(sample_size,target_dist.shape[1]))
        sample = mu_hat + sigma_hat*sample

        # Calculate Loss

        # sample case
        MMD_xx_case = MMD_equal_case(sample, device, b)
        MMD_xy_case = MMD_mixed_case(sample,target_dist,device, b)
        #loss
        loss = MMD_xx_case + MMD_yy_case - MMD_xy_case

        
        #optimizer.zero_grad()
        # Calculate gradient
        loss.backward()
        
        # Take one SGD step
        optimizer.step()

        mu_hat_estimates.append(mu_hat.detach().clone())
        sigma_hat_estimates.append(sigma_hat.detach().clone())
        MMD_values.append(loss.detach().clone())
        b_values.append(b)
        times.append(time.time()-start_time)


        
        if epoch_print_size:
            if epoch%epoch_print_size==0:
                print("epoch: ",epoch," loss=",loss)
            
    mu_hat_estimates = torch.stack(mu_hat_estimates).detach().numpy().reshape([nr_iterations,])
    sigma_hat_estimates = torch.stack(sigma_hat_estimates).detach().numpy().reshape([nr_iterations,])
    MMD_values = torch.stack(MMD_values).detach().numpy()

    df_results = pd.DataFrame({"mu_hat": mu_hat_estimates, 
              "sigma_hat": sigma_hat_estimates,
              "MMD": MMD_values,
              "b": b_values,
              "time": times})
    
    return df_results

def training_loop_multi_logist(alpha_hat, target_dist, nr_iterations , sample_size, device, b, optimizer, epoch_print_size=500, b_update=0):
   
    alpha_hat_estimates = []
    MMD_values = []
    b_values = []

    times = []
    start_time = time.time()

    #the MMD yy case is a constant.
    # no need to recalculate this for every epoch
    MMD_yy_case = MMD_equal_case(target_dist,device,b)
    
    for epoch in np.arange(nr_iterations):

        #iterative update of bandwidth value
        if b_update:

            if epoch % b_update == 0:

                params = {"alpha": alpha_hat.item()}
                
                
                b = calc_b_heuristic(target_dist, sample_size, "logistic", device, params).item()

                MMD_yy_case = MMD_equal_case(target_dist,device,b)


        # Empty gradient
        optimizer.zero_grad()

        # Sample from the generator
        sample = sample_multivariate_logistic(sample_size, target_dist.shape[1], alpha_hat, device)
        # Calculate Loss

        # sample case
        MMD_xx_case = MMD_equal_case(sample, device, b)
        MMD_xy_case = MMD_mixed_case(sample,target_dist,device, b)
        #loss
        loss = MMD_xx_case + MMD_yy_case - MMD_xy_case

        
        #optimizer.zero_grad()
        # Calculate gradient
        loss.backward()
        
        # Take one SGD step
        optimizer.step()


        # this is supposed to ensure that alpha_hat stays in the domain after optimization
        with torch.no_grad():  # Temporarily disable gradient tracking
            if alpha_hat <= 0:
                alpha_hat.copy_(torch.tensor(0.01))
            elif alpha_hat >= 1:
                alpha_hat.copy_(torch.tensor(0.99))


        alpha_hat_estimates.append(alpha_hat.detach().clone())
        MMD_values.append(loss.detach().clone())
        b_values.append(b)
        times.append(time.time()-start_time)


        

        if epoch_print_size:
            if epoch%epoch_print_size==0:
                print("epoch: ",epoch," loss=",loss)
            
    alpha_hat_estimates = torch.stack(alpha_hat_estimates).detach().numpy().reshape([nr_iterations,])
    MMD_values = torch.stack(MMD_values).detach().numpy()

    df_results = pd.DataFrame({"alpha_hat": alpha_hat_estimates,
              "MMD": MMD_values,
              "b": b_values,
              "time": times})
    
    return df_results


def MMD_mixed_case(x:torch.tensor,y:torch.tensor,device,b=1):

    """calculates the MMD between the sample and the target distribution 

    Args:
        x (torch.tensor): the current sample distribution 
        y (torch.tensor): the target distribution which we want to model
        device (_type_): device object for PyTorch
        b (int, optional): the bandwith value. Defaults to 1.

    Returns:
        _type_: the mixed case of the MMD which is a value
    """

    n = x.shape[0]
    m = y.shape[0]

    # calculates euclidean distance tensor
    cdist = torch.cdist(x,y)
 
    #calculate kernel values
    kernel_values = kernel_gauss_cdist(cdist,b).to(device)

  
    #get the sum
    sum = torch.sum(kernel_values) 

    #weighting the sum
    res = 2/(m*n) * sum

    return res

def MMD_equal_case(x:torch.tensor,device,b:float=1):
    """calculates the sample x sample and response x response case for the MMD

    Args:
        x (torch.tensor): the input tensor
        device (_type_): the PyTorch device object
        b (float, optional): the bandwith value . Defaults to 1.

    Returns:
        _type_: the MMD value for the equal case (a single value)
    """

    n = x.shape[0]

    # if x.isnan().any():

    #     print(x)
    
    cdist = torch.cdist(x,x,p=2).to(device)

    # if cdist.isnan().any():

    #     print(cdist)
    
    
    #calculate kernel values
    kernel_values = kernel_gauss_cdist(cdist,b)

    # if kernel_values.isnan().any():

    #     print(kernel_values)
    
    #get the diagonal
    diag_kernel_values = kernel_values.diagonal()

    #remove the diagonal j==i elements
    sum = torch.sum(kernel_values) - torch.sum(diag_kernel_values)
    
    res = 1/(n*(n-1)) * sum
    

    return res


def sample_multivariate_logistic(n:int, m:int, alpha:float, device)->torch.tensor:

    
        
    # Step 1: 
    # simulate from a positive stable distribution
    S = sample_PS(n, m, alpha, device)

    # Step 2:
    # sample random standard exponential variables independent of S
    W = torch.zeros([n,m]).exponential_(lambd=1).to(device)
    X = (S/W)**alpha

    return X


def sample_PS(n:int, m:int, alpha:torch.Tensor, device)-> torch.Tensor:

    # REMEMBER: unclear if S is rowwise identical. 
    # I assume so because of lack of index

    W = torch.zeros([n,1]).exponential_(1).to(device)

    U = torch.zeros([n,1]).uniform_(0,torch.pi).to(device)

    exponent = (1-alpha)/alpha

    comp1 = torch.sin((1-alpha)*U)/W

    comp2 = torch.sin(alpha*U)/(torch.sin(U)**(1-alpha))

    S_part_0 = comp1**exponent
    
    S =  S_part_0 * comp2

    # trying to prevent inf and 0 values

    # Define the max float and tiny value for torch.float64
    max_float = np.finfo(np.float64).max / 10000000
    tiny_float = np.finfo(np.float64).tiny * 10000000
    

    # 
    with torch.no_grad():
    # # Replace inf and -inf with max_float
        S = torch.where(torch.isinf(S), torch.tensor(max_float, dtype=torch.float64), S)

    # # Replace 0 with tiny_float
        S = torch.where(S == 0.0, torch.tensor(tiny_float, dtype=torch.float64), S)



    S = S.repeat(1,m)

    return S



def simu_px_brownresnick(no_simu, idx, N, trend, chol_mat):

    # Check the condition and raise an error if not met
    assert idx.numel() == 1 or idx.numel() == no_simu, "Length of idx must be 1 or no_simu"


    # Generate random normal matrix with N rows and no_simu columns
    # random component
    random_matrix = torch.randn(N, no_simu)
    #random_matrix = torch.ones(N, no_simu)

    # Perform matrix multiplication
    res = torch.mm(chol_mat.t(), random_matrix)

    # Apply trend and calculate exponentiated results
    if not isinstance(trend, torch.Tensor):
        trend = torch.tensor(trend)

    # Apply trend and calculate exponentiated results
    if trend.dim() == 1:
            #res = torch.exp((res.t() - trend).t())

            # Ensure trend is correctly broadcastable
            trend_expanded = trend.unsqueeze(1)  # Shape (N, 1)
            
            res = torch.exp((res - trend_expanded).t())
            

    else:
        #res = torch.exp((res.t() - trend[:, idx]).t())

        trend_expanded = trend[:, idx]  # Shape (N, no_simu)
        res = torch.exp((res - trend_expanded).t())

    # Normalize the results
    norm_factor = res[torch.arange(no_simu), idx]
    
    result = res / norm_factor.unsqueeze(1)
    
    return result

def sim_huesler_reiss(coord, Vario, device, loc=0., scale=1., shape=0., no_simu=1.):

    N = coord.shape[0]

    if isinstance(loc, float):

        loc = torch.tensor(np.repeat(loc, N))
        loc = loc.to(device)
     

    if isinstance(scale, float):

        scale = torch.tensor(np.repeat(scale, N))
        scale = scale.to(device)

    if isinstance(shape, float):

        shape = torch.tensor(np.repeat(shape, N))
        shape = shape.to(device)

    assert torch.all(scale > 1e-12), f"Not all elements in 'scale' {scale} are greater than 1e-12"

    #assert callable(vario), f" vario must be a function"

    # calculate the covariance matrix
    # Compute pairwise differences using broadcasting
    coord_i = coord.unsqueeze(1).expand(N, N, 2)  # Shape (N, N, 2)
    coord_j = coord.unsqueeze(0).expand(N, N, 2)  # Shape (N, N, 2)
    diff = coord_i - coord_j  # Shape (N, N, 2)

    # Apply the vario function
    vario_diff = Vario.vario(diff)  # Shape (N, N)

    # Apply the vario function to the original coordinates
    vario_coord = Vario.vario(coord)  # Shape (N,)

    # Compute the covariance matrix
    cov_mat = vario_coord.unsqueeze(1) + vario_coord.unsqueeze(0) - vario_diff
    # I guess we add this for numerical reasons?
    cov_mat = cov_mat + 1e-6

    # cholevski decomposition for upper triangular matrix

    chol_mat = torch.linalg.cholesky(cov_mat, upper=True)

    # get the trend which is the same as the difference
    trend = vario_diff

    # Initialize a zero matrix res with shape (no_simu, N)
    res = torch.zeros((no_simu, N))

    # Initialize a zero vector counter with length no_simu
    counter = torch.zeros(no_simu, dtype=torch.int)

    # draw exponential rv
    # random component
    poisson = torch.zeros([no_simu]).exponential_(lambd=1).to(device)
    #poisson = torch.ones([no_simu]).to(device)

    # get the loop termination condition
    ind = torch.tensor(np.repeat(True, no_simu))


    # actual algorithm => hopefully auto.diff can handle this loop

    while(ind.any()):
        
        
        n_ind = torch.sum(ind).item()
        counter[ind] = counter[ind] + 1


        # random component
        shift = torch.randint(0, N, (n_ind,), dtype=torch.int)


        # draw from the Hüsler Reis distribution
        proc = simu_px_brownresnick(n_ind, shift, N, trend, chol_mat)


        assert proc.shape == (n_ind, N), f"Shape of proc {proc.shape} does not match the expected dimensions {(n_ind, N)}"

        proc = N * proc / proc.sum(dim=1, keepdim=True)

        # maybe unsqueeze is an issue keep in mind
        res[ind, :] = torch.maximum(res[ind, :], proc / poisson[ind].unsqueeze(1))

        # create additional exponential term
        # random component
        exp_rv = torch.zeros(n_ind).exponential_(lambd=1).to(device)
        #exp_rv = torch.ones(n_ind).to(device)

        poisson[ind] = poisson[ind] + exp_rv

        ind = (N / poisson > res.min(dim=1).values)

        #print(f"{ind}")

    
    res_transformed = torch.where(
        torch.abs(shape) < 1e-12,
        torch.log(res) * scale.unsqueeze(0) + loc.unsqueeze(0),
        (1 / shape.unsqueeze(0)) * (res ** shape.unsqueeze(0) - 1) * scale.unsqueeze(0) + loc.unsqueeze(0)
        )


    # return {"res": res_transformed,
    #  	"counter": counter}

    return res_transformed


def sim_huesler_reis_ext(coord, Vario, device, loc=1., scale=1., shape=1., no_simu=1):

    assert isinstance(coord, torch.Tensor), f"coord must be a torch.tensor but is a {type(coord)}"

    N = coord.shape[0]

    if isinstance(loc, float):

        loc = torch.tensor(np.repeat(loc, N), device=device)
        
     

    if isinstance(scale, float):

        scale = torch.tensor(np.repeat(scale, N), device=device)
        

    if isinstance(shape, float):

        shape = torch.tensor(np.repeat(shape, N), device=device)
        

    assert torch.all(scale > 1e-12), f"all scale values must be bigger than 1e-12"


    # Compute covariance matrix using broadcasting
    coord_i = coord.unsqueeze(1)  # Shape (N, 1, d)
    coord_j = coord.unsqueeze(0)  # Shape (1, N, d)
    cov_matrix = Vario.vario(coord_i) + Vario.vario(coord_j) - Vario.vario(coord_i - coord_j)
    cov_matrix += 1e-6  # Add small constant for numerical stability

    # cholevski decomposition for upper triangular matrix
    chol_mat = torch.linalg.cholesky(cov_matrix, upper=True)


    # Initialize a zero matrix res with shape (no_simu, N)
    res = torch.zeros((no_simu, N))


    # Initialize a zero vector counter with length no_simu
    counter = torch.zeros(no_simu, dtype=torch.int)

    for k in range(N):

        # create additional exponential term
        # random component
        poisson = torch.zeros(no_simu).exponential_(lambd=1).to(device)
        #poisson = torch.ones(no_simu).to(device)

        trend = Vario.vario(coord -coord[k])

        while torch.any(1 / poisson > res[:, k]):

            ind = 1 / poisson > res[:, k]

            n_ind = ind.sum().item()
            idx = torch.arange(no_simu)[ind]
            counter[ind] += 1

            proc = simu_px_brownresnick(no_simu=n_ind, idx=torch.tensor([k]), N=N, trend=trend, chol_mat=chol_mat)

            assert proc.shape == (n_ind, N), f"Shape of proc {proc.shape} does not match the expected dimensions {(n_ind, N)}"


            if k == 0:

                ind_upd = torch.tensor(np.repeat(True, n_ind))
            else:

                #print([proc[i, :k] for i in range(n_ind)])
                ind_upd = torch.tensor([torch.all(1 / poisson[idx[i]] * proc[i, :k] <= res[idx[i], :k]) for i in range(n_ind)])

            if ind_upd.any():
                idx_upd = idx[ind_upd]
                res[idx_upd, :] = torch.maximum(res[idx_upd, :], 1 / poisson[idx_upd].unsqueeze(1) * proc[ind_upd, :])

            #this is random
            poisson[ind] = poisson[ind] + torch.zeros(n_ind).exponential_(lambd=1)
            #poisson[ind] = poisson[ind] + torch.ones(n_ind)

    # print(torch.abs(shape) < 1e-12)
    # print(torch.log(res))
    # print(scale.unsqueeze(0))
    # print(loc.unsqueeze(0))
    
    # Apply final transformation
    res_transformed = torch.where(
        torch.abs(shape) < 1e-12,
        torch.log(res) * scale.unsqueeze(0) + loc.unsqueeze(0),
        (1 / shape.unsqueeze(0)) * (res ** shape.unsqueeze(0) - 1) * scale.unsqueeze(0) + loc.unsqueeze(0)
    )

    # res_alt = torch.zeros(res.shape[0], res.shape[1])

    # for i in range(res.shape[1]):

    #     if torch.abs(shape[i])< 1e-12:
    #         res_alt[:, i] = torch.log(res[:,i]) * scale[i] + loc[i]

    #     else:
    #         res_alt[:,i] = (1 / shape[i]) * (res[:,i] ** shape[i] - 1) * scale[i] + loc[i]

    return res_transformed


def sim_huesler_reis_ext_safety(coord, vario, device, loc=1., scale=1., shape=1., no_simu=1):

    assert isinstance(coord, torch.Tensor), f"coord must be a torch.tensor but is a {type(coord)}"

    N = coord.shape[0]

    if isinstance(loc, float):

        loc = torch.tensor(np.repeat(loc, N), device=device)
        
     

    if isinstance(scale, float):

        scale = torch.tensor(np.repeat(scale, N), device=device)
        

    if isinstance(shape, float):

        shape = torch.tensor(np.repeat(shape, N), device=device)
        

    assert torch.all(scale > 1e-12), f"all scale values must be bigger than 1e-12"

    assert callable(vario), f"vario must be a function" 

    # Compute covariance matrix using broadcasting
    coord_i = coord.unsqueeze(1)  # Shape (N, 1, d)
    coord_j = coord.unsqueeze(0)  # Shape (1, N, d)
    cov_matrix = vario(coord_i) + vario(coord_j) - vario(coord_i - coord_j)
    cov_matrix += 1e-6  # Add small constant for numerical stability

    # cholevski decomposition for upper triangular matrix
    chol_mat = torch.linalg.cholesky(cov_matrix, upper=True)


    # Initialize a zero matrix res with shape (no_simu, N)
    res = torch.zeros((no_simu, N))


    # Initialize a zero vector counter with length no_simu
    counter = torch.zeros(no_simu, dtype=torch.int)

    for k in range(N):

        # create additional exponential term
        # random component
        poisson = torch.zeros(no_simu).exponential_(lambd=1).to(device)
        #poisson = torch.ones(no_simu).to(device)

        trend = vario(coord -coord[k])

        while torch.any(1 / poisson > res[:, k]):
            ind = 1 / poisson > res[:, k]
            n_ind = ind.sum().item()
            idx = torch.arange(no_simu)[ind]
            counter[ind] += 1

            proc = simu_px_brownresnick(no_simu=n_ind, idx=torch.tensor([k]), N=N, trend=trend, chol_mat=chol_mat)
            
            assert proc.shape == (n_ind, N), f"Shape of proc {proc.shape} does not match the expected dimensions {(n_ind, N)}"


            if k == 1:

                ind_upd = torch.tensor(np.repeat(True, n_ind))
            else:
                ind_upd = torch.tensor([torch.all(1 / poisson[idx[i]] * proc[i, :k] <= res[idx[i], :k]) for i in range(n_ind)])

            if ind_upd.any():
                idx_upd = idx[ind_upd]
                res[idx_upd, :] = torch.maximum(res[idx_upd, :], 1 / poisson[idx_upd].unsqueeze(1) * proc[ind_upd, :])

            #this is random
            poisson[ind] = poisson[ind] + torch.zeros(n_ind).exponential_(lambd=1)
            #poisson[ind] = poisson[ind] + torch.ones(n_ind)

    
    # Apply final transformation
    res_transformed = torch.where(
        torch.abs(shape) < 1e-12,
        torch.log(res) * scale.unsqueeze(0) + loc.unsqueeze(0),
        (1 / shape.unsqueeze(0)) * (res ** shape.unsqueeze(0) - 1) * scale.unsqueeze(0) + loc.unsqueeze(0)
    )

    return res_transformed
     


def create_centered_grid(size):
    """
    Create a centered grid with the given size.
    
    Args:
    size (int): The size of the grid (e.g., 2 for a 2x2 grid, 3 for a 3x3 grid, etc.).
    
    Returns:
    torch.Tensor: A tensor of shape (size*size, 2) representing the grid coordinates.
    """
    # Generate linear space from -1 to 1 with the specified size
    linear_space = torch.linspace(-1, 1, size)
    
    # Create the meshgrid from the linear space
    x, y = torch.meshgrid(linear_space, linear_space, indexing='ij')
    
    # Combine x and y coordinates into a single tensor and reshape it to the desired format
    grid = torch.stack([x, y], dim=-1).reshape(-1, 2)
    
    return grid


def plot_grid(grid):
    """
    Plot the grid points using a scatter plot.
    
    Args:
    grid (torch.Tensor): The grid tensor of shape (n*n, 2).
    title (str): The title of the plot.
    """
    # Convert the tensor to a numpy array for plotting
    grid_np = grid.numpy()
    
    grid_size = np.sqrt(grid.shape[0]).item()

    # Create the scatter plot
    plt.scatter(grid_np[:, 0], grid_np[:, 1], marker='o')
    plt.title(f"{grid_size}x{grid_size} Grid")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



class Vario:
     
    def __init__(self, alpha, p):
        
        self._alpha = alpha
        self._p = p

    @property
    def alpha(self):
         
         return self._alpha
    
    @alpha.setter
    def alpha(self, new_alpha):
         
         self._alpha = new_alpha

    @property
    def p(self):
         
        return self._p
    
    @p.setter
    def p(self, new_p):
         
         self._p = new_p



    def __str__(self):
        return f"Vario(alpha={self._alpha}, p={self._p})"
    
    def __repr__(self):
        return f"Vario(alpha={self._alpha}, p={self._p})"
    
    
    def vario(self,x):
         
         norm = self._alpha * torch.sqrt(torch.sum(x**2, dim=-1))**self._p

         return norm


def training_loop_huesler_reis(Vario, target_dist, grid, nr_iterations , sample_size, device, b, optimizer, epoch_print_size=500, b_update=0):
   
    alpha_hat_estimates = []
    p_hat_estimates = []
    MMD_values = []
    b_values = []

    times = []
    start_time = time.time()

    #the MMD yy case is a constant.
    # no need to recalculate this for every epoch
    MMD_yy_case = MMD_equal_case(target_dist,device,b)
    
    for epoch in np.arange(nr_iterations):

        #iterative update of bandwidth value
        if b_update:

            if epoch % b_update == 0:

                params = {"Vario": Vario,
                          "grid": grid}
                

                b = calc_b_heuristic(target_dist, sample_size, "huesler_reis", device, params).item()

                MMD_yy_case = MMD_equal_case(target_dist,device,b)


        
        # Empty gradient
        optimizer.zero_grad()

        # Sample from the generator
        sample = sim_huesler_reis_ext(grid, Vario, device, no_simu=sample_size)
        # Calculate Loss

        # sample case
        MMD_xx_case = MMD_equal_case(sample, device, b)
        MMD_xy_case = MMD_mixed_case(sample,target_dist,device, b)
        #loss
        loss = MMD_xx_case + MMD_yy_case - MMD_xy_case

        # Calculate gradient
        loss.backward()
        
       
        optimizer.step()
  

        
        with torch.no_grad():
            if Vario.alpha <= 0:
                Vario.alpha.copy_(torch.tensor(0.01))
                  
            elif Vario.p <= 0:
                 Vario.p.copy_(torch.tensor(0.01))
            
            elif Vario.p > 2:
                 Vario.p.copy_(torch.tensor(1.99))


        alpha_hat_estimates.append(Vario.alpha.detach().clone())
        p_hat_estimates.append(Vario.p.detach().clone())
        MMD_values.append(loss.detach().clone())
        b_values.append(b)
        times.append(time.time()-start_time)


        if epoch_print_size:
            if epoch%epoch_print_size==0:
                print("epoch: ",epoch," loss=",loss)
            
    alpha_hat_estimates = torch.stack(alpha_hat_estimates).detach().numpy().reshape([nr_iterations,])
    p_hat_estimates = torch.stack(p_hat_estimates).detach().numpy().reshape([nr_iterations,])
    MMD_values = torch.stack(MMD_values).detach().numpy()

    df_results = pd.DataFrame({"alpha_hat": alpha_hat_estimates,
                               "p_hat": p_hat_estimates,
                                "MMD": MMD_values,
                                "b": b_values,
                                "time": times})
    
    return df_results