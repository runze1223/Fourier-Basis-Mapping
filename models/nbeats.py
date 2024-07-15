# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:20:30 2022

@author: dell
"""
import math
import numpy as np
import torch as t
import torch.nn as nn
from typing import Tuple
from layers.tcn import TemporalConvNet

import matplotlib.pyplot as plt




class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial, backcast_size, forecast_size,variation):
        super(TrendBasis,self).__init__()
        polynomial_size = degree_of_polynomial + 1
          
        total_grid=t.tensor(np.concatenate([np.power(np.arange(forecast_size+backcast_size, dtype=np.float) / (forecast_size+backcast_size), i)[None, :]
                                    for i in range(polynomial_size)]), dtype=t.float32)**0.75
        backcast_template=total_grid[:,:backcast_size]
        forecast_template=total_grid[:,backcast_size:]
        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:      
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bkp,pt->bkt', theta, self.backcast_basis)
        forecast = t.einsum('bkp,pt->bkt', theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics, backcast_size, forecast_size,degree_of_polynomial,variation):
        super(SeasonalityBasis,self).__init__()

        
        frequency = np.append(np.zeros(1, dtype=np.float32),
                                        np.arange(harmonics, harmonics / 2 * (forecast_size+backcast_size),
                                                    dtype=np.float32) / harmonics)[None, :]
        total_grid = -2 * np.pi * (
                np.arange(backcast_size+forecast_size, dtype=np.float32)[:, None] / (forecast_size+backcast_size)) * frequency

        total_grid2 = -2 * np.pi * (
                np.arange(backcast_size+forecast_size, dtype=np.float32)[:, None] / (forecast_size+backcast_size)) * frequency- -0.25 * np.pi* frequency

        self.backcast_size  =  backcast_size   

        backcast_grid=total_grid[:backcast_size,:]
        forecast_grid=total_grid[backcast_size:,:]
        backcast_grid2=total_grid2[:backcast_size,:]
        forecast_grid2=total_grid2[backcast_size:,:]
        
        backcast_cos_template = t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32)
        backcast_sin_template = t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32)
        backcast_cos_template2 = t.tensor(np.transpose(np.cos(backcast_grid2)), dtype=t.float32)
        backcast_sin_template2 = t.tensor(np.transpose(np.sin(backcast_grid2)), dtype=t.float32)
        backcast_template = t.cat([backcast_cos_template, backcast_sin_template,backcast_cos_template2, backcast_sin_template2], dim=0)
        forecast_cos_template = t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32)
        forecast_sin_template = t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32)
        forecast_cos_template2 = t.tensor(np.transpose(np.cos(forecast_grid2)), dtype=t.float32)
        forecast_sin_template2 = t.tensor(np.transpose(np.sin(forecast_grid2)), dtype=t.float32)
        forecast_template = t.cat([forecast_cos_template, forecast_sin_template,forecast_cos_template2, forecast_sin_template2], dim=0)
        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bkp,pt->bkt', theta, self.backcast_basis)
        forecast = t.einsum('bkp,pt->bkt', theta, self.forecast_basis)
        return backcast, forecast 

       
class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, theta_n_dim, basis: nn.Module,
                 n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, activation,variation,channel):
        """
        """
        super(NBeatsBlock,self).__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden
        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.variation=variation
        self.channel=channel
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):
            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1]))
            # hidden_layers.append(self.activations[activation])
            if i+1<n_layers:
                if self.batch_normalization:
                    hidden_layers.append(nn.InstanceNorm1d(num_features=theta_n_hidden[i+1]))
                if self.dropout_prob>0:
                    hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        layers = hidden_layers
        self.layers = nn.Sequential(*layers)

        hidden_layers = []
        for i in range(n_layers):
            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1]))     
            hidden_layers.append(self.activations[activation])
            if i<n_layers:
                if self.batch_normalization:
                    # hidden_layers.append(nn.InstanceNorm1d(num_features=theta_n_hidden[i+1]))
                    hidden_layers.append(nn.InstanceNorm1d(num_features=16))
                # if self.dropout_prob>0:
                #     hidden_layers.append(nn.Dropout(p=self.dropout_prob))    
        layers = hidden_layers
        self.layers2 = nn.Sequential(*layers)
        self.output_layer=nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)
        self.basis = basis
        
    def sample(self, mu, log_var):
        std = t.exp(0.5*log_var)
        eps = t.randn_like(std)
        return mu + eps*std
    
    def vae_loss(self,mu, logvar):
        KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return KL_loss
        

    def forward(self, insample_y) -> Tuple[t.Tensor, t.Tensor]:
        # Compute local projection weights and 
        
        if self.variation==True:
            if self.training:
                mu = self.layers(insample_y)
                log_var = self.layers2(insample_y)
                ouput = self.sample(mu, log_var)
                kl_divergence=self.vae_loss(mu,log_var)
            else:
                mu = self.layers(insample_y)
                ouput = mu
                log_var = None
                kl_divergence=t.tensor([0]).float().cuda()
                
            theta=self.output_layer(ouput)
            
        else:
            ouput=self.layers(insample_y)
            theta=self.output_layer(ouput)

        backcast, forecast = self.basis(theta)
        if  self.variation==True:
            return backcast, forecast,kl_divergence,theta
        else:
            return backcast, forecast
            


class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self,  x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
                 n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation,variation,channel):    
        super(NBeats,self).__init__()
        
        self.input_size=x_t_n_inputs
        self.n_hidden=theta_n_hidden
        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob
        self.activation = activation
        self.n_layers=n_layers
        self.output_size=output_size
        self.n_polynomials=n_polynomials
        self.n_harmonics=n_harmonics
        self.variation=variation
        self.channel=channel
        block_list=[]
        self.nbeats_block = NBeatsBlock(x_t_n_inputs = self.input_size,
                                    x_s_n_inputs =0,
                                    x_s_n_hidden= self.x_s_n_hidden,
                                    theta_n_dim=(self.n_polynomials + 1),
                                    basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                            backcast_size=self.input_size,
                                                            forecast_size=self.output_size,
                                                            variation=self.variation).cuda(),
                                    n_layers=self.n_layers,
                                    theta_n_hidden=[ 2*(self.n_polynomials + 1)],
                                    include_var_dict=self.include_var_dict,
                                    t_cols=self.t_cols,
                                    batch_normalization=self.batch_normalization,
                                    dropout_prob=self.dropout_prob_theta,
                                    activation=self.activation,
                                    variation=self.variation,
                                    channel=self.channel)
        block_list.append(self.nbeats_block)
        self.nbeats_block2 = NBeatsBlock(x_t_n_inputs = self.input_size,
                x_s_n_inputs = 0,
                x_s_n_hidden= self.x_s_n_hidden,
                theta_n_dim= 4 * int(
                        np.ceil(self.n_harmonics / 2 * (self.output_size+self.input_size)) - (self.n_harmonics - 1)),
                basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                        backcast_size=self.input_size,
                                        forecast_size=self.output_size,
                                        degree_of_polynomial=self.n_polynomials,
                                        variation=self.variation).cuda(),
                n_layers=self.n_layers,
                theta_n_hidden= [4 * int(
                        np.ceil(self.n_harmonics / 2 * (self.output_size+self.input_size)) - (self.n_harmonics - 1))],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=self.batch_normalization,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation,
                variation=self.variation,
                channel=self.channel)
        block_list.append(self.nbeats_block2)
        self.blocks= block_list

    def forward(self, insample_y):
        insample_y= insample_y.permute(0,2,1)
        residuals = insample_y
        forecastbase = insample_y[:,:, -1:] # Level with Naive1
        residuals=residuals-forecastbase
        forecast=forecastbase
        forecast_back=forecastbase

        block_forecasts = []
        thetas=[]
        if self.variation==True:
            kl_divergence_total=[]
            for i, block in enumerate(self.blocks):
                backcast, block_forecast,kl_divergence,theta = block(insample_y=residuals) 
                forecast_back = forecast_back  + backcast
                residuals = (residuals - backcast) 
                forecast = forecast + block_forecast
                kl_divergence_total.append(kl_divergence)
                thetas.append(theta)
            kl_divergence_total=t.stack(kl_divergence_total)


            return forecast, forecast_back, kl_divergence_total
            
        else:
            for i, block in enumerate(self.blocks):
                backcast, block_forecast = block(insample_y=residuals)
            
                forecast_back = forecast_back + backcast
                residuals = (residuals - backcast) 
                forecast = forecast + block_forecast
                block_forecasts.append(block_forecast)
            return forecast, forecast_back

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        
        self.input_size=configs.seq_len
        self.n_hidden=configs.theta_n_hidden
        self.x_s_n_inputs = 0
        self.x_s_n_hidden = 0
        self.include_var_dict = None
        self.t_cols = None
        self.batch_normalization = configs.batch_normalization
        self.dropout_prob_theta = configs.dropout_prob
        self.activation = configs.activation_nbeats
        self.n_layers=configs.n_layers
        self.output_size=configs.pred_len
        self.n_polynomials=configs.n_polynomials
        self.n_harmonics=configs.n_harmonics
        self.channel=configs.enc_in
        self.individual=configs.individual
        self.variation=configs.variation
        self.decom=1

        self.activations = {'relu': nn.ReLU(),
                    'softplus': nn.Softplus(),
                    'tanh': nn.Tanh(),
                    'selu': nn.SELU(),
                    'lrelu': nn.LeakyReLU(),
                    'prelu': nn.PReLU(),
                    'sigmoid': nn.Sigmoid()}

        if self.individual==True:
            self.nbeats_together = nn.ModuleList()
            for i in range(self.channel):
                self.nbeats_together.append(NBeats(self.input_size, self.x_s_n_inputs,self.x_s_n_hidden,self.n_polynomials, self.n_harmonics,
                              self.n_layers, self.n_hidden, self.include_var_dict, self.t_cols, self.batch_normalization, self.dropout_prob_theta, self.output_size, self.activation,self.variation,self.channel))
        else:
            self.nbeats=NBeats(self.input_size, self.x_s_n_inputs,self.x_s_n_hidden,self.n_polynomials, self.n_harmonics,
                          self.n_layers, self.n_hidden, self.include_var_dict, self.t_cols, self.batch_normalization, self.dropout_prob_theta, self.output_size, self.activation,self.variation,self.channel)   

        
    def forward(self, x_enc):
        if self.individual==True:
            if self.variation==True:
                forecast_total=[]
                forecastback_total=[]
                kl_divergence_total=[]
                for i in range(self.channel):  
                    x_input=x_enc[:,:,i]
                    forecast,forecastback, kl_divergence=self.nbeats_together[i](x_input)
                    forecast_total.append(forecast)
                    forecastback_total.append(forecastback)
                    kl_divergence_total.append(kl_divergence)
                forecast_total=t.stack(forecast_total,axis=2)
                forecastback_total=t.stack(forecastback_total,axis=2)
                kl_divergence_total=t.mean(t.stack(kl_divergence_total,axis=0))
            else:
                forecast_total=[]
                forecastback_total=[]
                for i in range(self.channel):            
                    x_input=x_enc[:,:,i]
                    forecast,forecastback=self.nbeats_together[i](x_input)
                    forecast_total.append(forecast)
                    forecastback_total.append(forecastback)
                forecast_total=t.stack(forecast_total,axis=2)
                forecastback_total=t.stack(forecastback_total,axis=2)
            output=forecast_total
            if self.variation==True:
                return output,forecastback_total,kl_divergence_total# [B, L, D],
            else: 
                return output# [B, L, D],
        else:
            if self.variation==True:
                x_input=x_enc
                forecast,forecastback, kl_divergence=self.nbeats(x_input)
                kl_divergence=t.mean(kl_divergence,axis=0)
            else:
                x_input=x_enc
                forecast,forecastback=self.nbeats(x_input)
            forecast=forecast.permute(0,2,1)
            forecastback=forecastback.permute(0,2,1)
            
            if self.variation==True:
                return forecast,forecastback, kl_divergence# [B, L, D],
            else: 
                return forecast# [B, L, D],
