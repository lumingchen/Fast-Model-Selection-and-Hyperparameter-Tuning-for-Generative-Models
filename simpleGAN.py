import torch
import torch.nn as nn

from utils import (
    distributional_generalized_sliced_wasserstein_distance,
    distributional_sliced_wasserstein_distance,
    generalized_sliced_wasserstein_distance,
    max_generalized_sliced_wasserstein_distance,
    sliced_wasserstein_distance,
    max_sliced_wasserstein_distance,
    augmented_sliced_wassersten_distance
)
from torch import optim
from TransformNet import TransformNet, Mapping


class Decoder(nn.Module):
    def __init__(self, size, latent_size):
        super(Decoder, self).__init__()
        self.size = size

        main = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, size),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):
    def __init__(self, size, hidden_size):
        super(Discriminator, self).__init__()

        self.main1 = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, hidden_size),
            nn.ReLU(True)
#             nn.Tanh()
        )
        self.main2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.main1(x)
        y = self.main2(h).view(x.shape[0], -1)
#         output = self.main(inputs)
#         return output.view(-1)
        return y, h
    
    


class simpleAutoencoder(nn.Module):
    def __init__(self, size, hidden_size, latent_size, device, model_type = None):
        super(simpleAutoencoder, self).__init__()
        self.size = size
        self.latent_size = latent_size
        self.device = device
        self.decoder = Decoder(size, latent_size)
        
        self.model_type = model_type
        self.num_projection = 1000
        self.gsw = None
        self.g_function = None
        self.r = 1
        self.max_iter = 10
        self.lam = 1
        self.tnet = TransformNet(hidden_size).to(device)
        self.op_tnet = None
#         optim.Adam(self.tnet.parameters(), lr=0.005, betas=(0.5, 0.999))
        self.phi = Mapping(hidden_size).to(device) 
        self.phi_op = None
#         optim.Adam(self.phi.parameters(), lr=0.005, betas=(0.5, 0.999))
        self.dir = None
        self.optimizer = None

            
    def compute_loss_SWD(self, discriminator, optimizer, minibatch, rand_dist, num_projection, p=2):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)
        
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)

        _swd = sliced_wasserstein_distance(
            data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1), num_projection, p, self.device
        )

        return _swd

    def compute_loss_MGSWNN(self, discriminator, optimizer, minibatch, rand_dist, gsw, max_iter, p=2):       
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)        
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        gswd = gsw.max_gsw(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1), iterations=max_iter)
        return gswd  
        
    def compute_loss_GSWNN(self, discriminator, optimizer, minibatch, rand_dist, gsw, p=2):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)  
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        gswd = gsw.gsw(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1))
        return gswd  

    def compute_loss_GSWD(self, discriminator, optimizer, minibatch, rand_dist, g_function, r, num_projection, p=2):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)  
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)

        _gswd = generalized_sliced_wasserstein_distance(
            data.view(data.shape[0], -1),
            data_fake.view(data.shape[0], -1),
            g_function,
            r,
            num_projection,
            p,
            self.device,
        )
        return _gswd

        
    def compute_lossDGSWD(self, discriminator, optimizer, minibatch, rand_dist, num_projections, tnet, op_tnet, g, r, p=2, max_iter=100, lam=1):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)  
        
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        _dswd = distributional_generalized_sliced_wasserstein_distance(
            data.view(data.shape[0], -1),
            data_fake.view(data.shape[0], -1),
            num_projections,
            tnet,
            op_tnet,
            g,
            r,
            p,
            max_iter,
            lam,
            self.device,
        )
        return _dswd


    def compute_loss_MSWD(self, discriminator, optimizer, minibatch, rand_dist, gsw, max_iter):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)         
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        _mswd = gsw.max_gsw(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1), iterations=max_iter)
        return _mswd

        

    def compute_loss_MGSWD(self, discriminator, optimizer, minibatch, rand_dist, g, r, p=2, max_iter=100):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake) 
        
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        _mswd = max_generalized_sliced_wasserstein_distance(
            data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1), g, r, p, max_iter
        )
        return _mswd

    
    def compute_lossASWD(self, discriminator, optimizer, minibatch, rand_dist, num_projections, phi, phi_op, p=2, max_iter=10, lam=1):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake) 
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        _aswd = augmented_sliced_wassersten_distance(data.view(data.shape[0], -1),
                                                            data_fake.view(data.shape[0], -1), num_projections, phi,
                                                            phi_op,
                                                            p, max_iter, lam,
                                                            self.device)
        return _aswd
    
    
    def compute_lossDSWD(self, discriminator, optimizer, minibatch, rand_dist, num_projections, tnet, op_tnet, p=2, max_iter=100, lam=1):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward()
        optimizer.step()
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake) 
        
#         data = minibatch.to(self.device)
#         z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
#         data_fake = self.decoder(z_prior)
        _dswd = distributional_sliced_wasserstein_distance(
            data.view(data.shape[0], -1),
            data_fake.view(data.shape[0], -1),
            num_projections,
            tnet,
            op_tnet,
            p,
            max_iter,
            lam,
            self.device,
        )
        return _dswd
    
    
    def compute_loss(self, discriminator, optimizer, minibatch, rand_dist):
        if self.model_type == 'SWD':
            return self.compute_loss_SWD(discriminator, optimizer, minibatch, rand_dist, self.num_projection, 2)
        elif self.model_type == 'GSWNN':
            return self.compute_loss_GSWNN(discriminator, optimizer, minibatch, rand_dist, self.gsw, 2)
        elif self.model_type == 'MGSWNN':
            return self.compute_loss_MGSWNN(discriminator, optimizer, minibatch, rand_dist, self.gsw, self.max_iter, 2)
        elif self.model_type == 'GSWD':
            return self.compute_loss_GSWD(discriminator, optimizer, minibatch, rand_dist, self.g_function, self.r, self.num_projection, 2)
        elif self.model_type == 'DGSWD':
            return self.compute_lossDGSWD(discriminator, optimizer, minibatch, rand_dist, self.num_projection, self.tnet, self.op_tnet, self.g_function, self.r, 2, self.max_iter, self.lam)
        elif self.model_type == 'MSWD':
            return self.compute_loss_MSWD(discriminator, optimizer, minibatch, rand_dist, 2, self.max_iter)
        elif self.model_type == 'MGSWD':
            return self.compute_loss_MGSWD(discriminator, optimizer, minibatch, rand_dist, self.g_function, self.r, 2, self.max_iter)
        elif self.model_type == 'ASWD':
            return self.compute_lossASWD(discriminator, optimizer, minibatch, rand_dist, self.num_projection, self.phi, self.phi_op, 2, self.max_iter, self.lam)
        elif self.model_type == 'DSWD':
            return self.compute_lossDSWD(discriminator, optimizer, minibatch, rand_dist, self.num_projection, self.tnet, self.op_tnet, 2, self.max_iter, self.lam)
            
