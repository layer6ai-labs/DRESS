"""
Meta-GMVAE Code Adapted from https://github.com/db-Lee/Meta-GMVAE
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GMVAE_Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(GMVAE_Encoder, self).__init__()
        self.last_hidden_size = 2*2*hidden_size
        self.encoder = nn.Sequential(
            # -1, hidden_size, 14, 14
            nn.Conv2d(1, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            # -1, hidden_size, 7, 7
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            # -1, 2*hidden_size, 4, 4
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            # -1, 2*hidden_size, 2, 2
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True)
        )

    def forward(self, x):
        # -1, hidden_size, 2, 2
        h = self.encoder(x)
        # -1, hidden_size*2*2
        h = h.view(-1, self.last_hidden_size)
        return h

class GMVAE_Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(GMVAE_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.Sequential(
            # -1, hidden_size, 4, 4
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 7, 7
            nn.ConvTranspose2d(hidden_size, hidden_size, 3, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 14, 14
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, 1, 28, 28
            nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1),
            nn.Sigmoid()
        )        

    def forward(self, h):
        # -1, hidden_size, 2, 2
        h = h.view(-1, self.hidden_size, 2, 2)
        # -1, 1, 28, 28
        x = self.decoder(h)
        return x

# The below code is adapted from github.com/juho-lee/set_transformer/blob/master/modules.py 
class GMVAE_MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(GMVAE_MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        #O = O + F.elu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class GMVAE_SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(GMVAE_SAB, self).__init__()
        self.mab = GMVAE_MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class GMVAE(nn.Module):
    def __init__(
        self,
        hidden_size,
        component_size,   
        latent_size, 
        args
    ):
        super(GMVAE, self).__init__()
        
        self.input_size = args.imgSizeToEncoder
        self.unsupervised_em_iters = 5
        self.semisupervised_em_iters = 5
        self.fix_pi = False
        self.hidden_size = hidden_size
        self.last_hidden_size = 2*2*hidden_size
        self.component_size = component_size
        self.latent_size = latent_size
        self.train_mc_sample_size = 32
        self.test_mc_sample_size = 32

        self.encoder = GMVAE_Encoder(
            hidden_size=hidden_size             
        )

        self.q_z_given_x_net = nn.Sequential(  
            GMVAE_SAB(
                dim_in=self.last_hidden_size, 
                dim_out=self.last_hidden_size, 
                num_heads=4,
                ln=False
            ),
            GMVAE_SAB(
                dim_in=self.last_hidden_size, 
                dim_out=self.last_hidden_size, 
                num_heads=4, 
                ln=False
            ),          
            nn.Linear(self.last_hidden_size, 2*self.hidden_size)
        )

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.last_hidden_size, self.last_hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.last_hidden_size, self.last_hidden_size),
            nn.ELU(inplace=True),
        )

        self.decoder = GMVAE_Decoder(
            hidden_size=hidden_size          
        )

        self.rec_criterion = nn.BCELoss(reduction='sum')
        self.register_buffer('log_norm_constant', torch.tensor(-0.5 * np.log(2 * np.pi)))
        self.register_buffer('uniform_pi', torch.ones(self.component_size)/self.component_size)
        
    def reparametrize(self, mean, logvar, S=1):
        mean = mean.unsqueeze(1).repeat(1, S, 1)
        logvar = logvar.unsqueeze(1).repeat(1, S, 1)
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(mean)
        return eps.mul(std).add(mean)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]        
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1), 
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)
        
        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1), 
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)        
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        supervised_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_X
        ) 

        # batch_size, component_size, latent_size
        unsupervised_mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            unsupervised_X
        ) 
        mean = (supervised_mean + unsupervised_mean) / denominator

        supervised_X2 = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_X.pow(2.0)
        )

        supervised_X_mean = mean * torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_X
        )
        
        # batch_size, component_size, latent_size
        supervised_mean2 = supervised_N[:, :, None].repeat(1, 1, self.latent_size) * mean.pow(2.0)

        supervised_var = supervised_X2 - 2 * supervised_X_mean + supervised_mean2

        unsupervised_X2 = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            unsupervised_X.pow(2.0)
        )

        unsupervised_X_mean = mean * torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            unsupervised_X
        )
        
        # batch_size, component_size, latent_size
        unsupervised_mean2 = unsupervised_N[:, :, None].repeat(1, 1, self.latent_size) * mean.pow(2.0)
        unsupervised_var = unsupervised_X2 - 2 * unsupervised_X_mean + unsupervised_mean2
        var = (supervised_var + unsupervised_var) / denominator
        logvar = torch.log(var)

        return pi, mean, logvar

    def get_posterior(self, H, mc_sample_size=10):
        batch_size, sample_size = H.shape[:2]

        ## q z ##
        # batch_size, sample_size, latent_size
        q_z_given_x_mean, q_z_given_x_logvar = self.q_z_given_x_net(H).split(self.latent_size, dim=-1)
        # batch_size, sample_size, mc_sample_size, latent_size
        q_z_given_x = self.reparametrize(
            mean=q_z_given_x_mean.view(batch_size*sample_size, self.latent_size), 
            logvar=q_z_given_x_logvar.view(batch_size*sample_size, self.latent_size), 
            S=mc_sample_size
        ).view(batch_size, sample_size, mc_sample_size, self.latent_size)

        return q_z_given_x_mean, q_z_given_x_logvar, q_z_given_x

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]
        
        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)
        
        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z, 
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)
        
        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, self.input_size, self.input_size)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_z
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, mc_sample_size, latent_size
        q_z_given_x_mean, q_z_given_x_logvar, q_z_given_x = self.get_posterior(H, self.train_mc_sample_size)

        # p_z
        all_z = q_z_given_x.view(batch_size, -1, self.latent_size)
        p_z_given_psi = self.get_unsupervised_prior(z=all_z)
        # batch_size, component_size
        # batch_size, component_size, latent_size
        p_y_given_psi_pi, p_z_given_y_psi_mean = p_z_given_psi

        ## decode ##
        # batch_size*sample_size*mc_sample_size, latent_size
        H_rec = self.proj(q_z_given_x.view(-1, self.latent_size))
        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec
        ).view(batch_size, sample_size, self.train_mc_sample_size, self.input_size, self.input_size)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec, 
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        )/(batch_size*sample_size*self.train_mc_sample_size)

        ## kl loss ##
        # batch_size, sample_size, mc_sample_size
        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z_given_x,
            q_z_given_x_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_given_x_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )
        
        # batch_size, sample_size, mc_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z_given_x[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(p_y_given_psi_pi[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))
        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        #kl_loss = torch.mean(log_qz.mean(dim=-1) - log_pz.mean(dim=-1))
        kl_loss = torch.mean(log_qz - log_pz)

        return rec_loss, kl_loss

    def prediction(self, X_tr, y_tr, X_te):
        batch_size, tr_sample_size = X_tr.shape[:2]
        te_sample_size = X_te.shape[1]

        # batch_size, tr_sample_size+te_sample_size, 1, 28, 28
        X = torch.cat([X_tr, X_te], dim=1)

        # encode
        H = self.encoder(
            X.view(-1, self.input_size, self.input_size)
        ).view(batch_size, tr_sample_size+te_sample_size, self.last_hidden_size)

        # q_z
        # batch_size, tr_sample_size+te_sample_size, mc_sample_size, latent_size
        _, _, q_z_given_x = self.get_posterior(H, self.test_mc_sample_size)
        
        ## p z ##
        # batch_size, te_sample_size*mc_sample_size, latent_size
        unsupervised_z = q_z_given_x[:, tr_sample_size:(tr_sample_size+te_sample_size), :, :].view(batch_size, te_sample_size*self.test_mc_sample_size, self.latent_size)
        # batch_size, tr_sample_size*mc_sample_size, latent_size
        supervised_z = q_z_given_x[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size*self.test_mc_sample_size, self.latent_size)        
        # batch_size, tr_sample_size*mc_sample_size
        y = y_tr.view(batch_size*tr_sample_size)[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size*self.test_mc_sample_size)
        # batch_size, tr_sample_size*mc_sample_size, component_size
        y = F.one_hot(y, self.component_size).float()
                
        # p_z
        p_z_given_psi = self.get_semisupervised_prior(
            unsupervised_z=unsupervised_z,
            supervised_z=supervised_z,
            y=y
        )

        # batch_size, component_size
        # batch_size, component_size, latent_size
        # batch_size, component_size, latent_size
        p_y_given_psi_pi, p_z_given_y_psi_mean, p_z_given_y_psi_logvar = p_z_given_psi

        # batch_size, te_sample_size, mc_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, te_sample_size, mc_sample_size, component_size, latent_size
            unsupervised_z.view(batch_size, te_sample_size, self.test_mc_sample_size, self.latent_size)[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1), 
            p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1),
            p_z_given_y_psi_logvar[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1)
        ) + torch.log(p_y_given_psi_pi[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

        # batch_size, sample_size, mc_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )
        # batch_size, sample_size
        y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

        return y_te_pred
