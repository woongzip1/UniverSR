from abc import ABC, abstractmethod

import torch
from torchdiffeq import odeint
from tqdm import tqdm

from universr.models.unet import ConditionalVectorFieldModel

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass
    
class Solver(ABC):
    # @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

class VectorFieldODE(ODE):
    def __init__(self, net:ConditionalVectorFieldModel) -> None:
        super().__init__()
        self.net = net
        
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(xt, t, y, **kwargs)
    
class CFGVectorFieldODE(ODE):
    """ For Classifier Free Guidance """
    def __init__(self, net:ConditionalVectorFieldModel, guidance_scale: float = 1.0) -> None:
        super().__init__()
        self.net = net
        self.guidance_scale = guidance_scale
        
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        guided_vector_field = self.net(xt, t, y, **kwargs)
        unguided_vector_field = self.net(xt, t, None, **kwargs)
        
        return (1-self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

class EulerSolver(Solver):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt,t, **kwargs) * h
    
class TorchDiffeqSolver(Solver):
    def __init__(self,
                 ode: ODE,
                 method: str = 'euler',
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 ):
        super().__init__()
        self.ode = ode
        self.method = method
        self.atol = atol
        self.rtol = rtol
    
    @torch.no_grad()
    def simulate(self, x_init: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        x_init: [B,C,H,W]
        ts: [N]
        return: final state [B,C,H,W]
        """        
        func = lambda t, x: self.ode.drift_coefficient(xt=x, t=t, **kwargs)
        
        xs = odeint(
                    func=func, 
                    y0=x_init, t=ts, 
                    method=self.method,
                    atol=self.atol, rtol=self.rtol) # [N,B,C,H,W]
        return xs[-1].clone()