from .base import Surrogate

import gc
import torch
import gpytorch
from tqdm import tqdm
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.models import AbstractVariationalGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

class DGP(Surrogate):
    def __init__(self, **kwargs):
        super(DGP, self).__init__()

        self.name         = 'DGP'
        self.model        = kwargs['model'] if 'model' in kwargs else None
        self.n_epochs     = kwargs['n_epochs'] if 'n_epochs' in kwargs else 200
        self.lr           = kwargs['lr'] if 'lr' in kwargs else 1e-1
        self.verbose      = kwargs['verbose'] if 'verbose' in kwargs else 1
        self.num_inducing = kwargs['num_inducing'] if 'num_inducing' in kwargs else 64
        self.hidden_size  = kwargs['hidden_size'] if 'hidden_size' in kwargs else 128
        self.actf         = kwargs['actf'] if 'actf' in kwargs else torch.tanh
        self.means        = kwargs['means'] if 'means' in kwargs else ['linear', 'linear']
        self.scale        = kwargs['scale'] if 'scale' in kwargs else True

        self.gen = torch.Generator() # for reproducibility
        self.gen.manual_seed(2208060503)
    
    def fit(self, X, y):
        
        del self.model
        gc.collect()

        self.model = DeepGPModel(
            in_dim       = X.shape[1],
            hidden_size  = self.hidden_size,
            num_inducing = self.num_inducing,
            actf         = self.actf,
            means        = self.means,
            gen          = self.gen
        )
        
        X = torch.tensor(X).float()
        y = torch.tensor(y).float().squeeze()
        
        if self.scale:
            self.y_mu, self.y_std = y.mean(), y.std()
            y = (y - self.y_mu) / (self.y_std + 1e-8)
        
        self.model.train()
        self.model.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        mll = DeepApproximateMLL(VariationalELBO(self.model.likelihood, self.model, num_data=y.shape[0]))

        if self.verbose > 1:
            pbar = tqdm(range(self.n_epochs), leave=False)
        else:
            pbar = range(self.n_epochs)
        
        for epoch in pbar:
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.mean().backward()
            optimizer.step()
            
            if self.verbose > 1:
                pbar.set_description(f'Epoch {epoch + 1}/{self.n_epochs} - Loss: {loss.mean().item():.3f}')

    
    def predict(self, X):

        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(torch.tensor(X).float()))
            mean = observed_pred.mean[0, 0, :]
            std = torch.sqrt(observed_pred.variance)[0, 0, :]
        
        if self.scale:
            # print(mean.shape, std.shape, self.y_mu.shape, self.y_std.shape)
            mean += self.y_mu
            std *= self.y_std
        
        return mean.numpy(), std.numpy()


class SingleLayerGP(AbstractVariationalGP):
    def __init__(self, inducing_points, mean='const'):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean() if mean == 'const' else  LinearMean(inducing_points.size(1))
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=inducing_points.size(1))) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, inducing_points, mean_type='constant'):
        if output_dims is None:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=input_dims
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGPModel(DeepGP):
    def __init__(self, in_dim, out_dim=None, hidden_size=16, num_inducing=22, layers=['deep', 'deep'], means=['constant', 'constant'], actf=torch.tanh, gen=None):
        super().__init__()
        
        self.actf = actf
        
        inducing_points = torch.rand(num_inducing, in_dim, generator=gen)
        output_inducing = torch.rand(num_inducing, hidden_size if layers[1]=='deep' else 1, generator=gen)
        
        self.input_layer = DeepLayer(in_dim, hidden_size, inducing_points, mean_type=means[0]) if layers[0]=='deep' else SingleLayerGP(inducing_points)
        self.output_layer =  DeepLayer(hidden_size, None, output_inducing, mean_type=means[1]) if layers[1]=='deep' else SingleLayerGP(output_inducing)
        
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        hidden_rep = self.input_layer(x).mean
        if self.actf is not None:
            hidden_rep = self.actf(hidden_rep)
        output = self.output_layer(hidden_rep)
        return output
