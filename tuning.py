import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import save_dmodel
from tqdm import tqdm
from gsw import GSW
from simpleGAN import Discriminator
import mmd



class Tuning(object): 
    def __init__(self, args, tarin_data, device):
        self.max_iter = args.epochs
        self.train_loader = DataLoader(TensorDataset(tarin_data), batch_size=args.train_size, drop_last=True)
        # self.val_loader = test_loader
        self.val_size = args.test_size
        self.lr = args.lr
        self.hidden_size = args.hidden_size
        self.device = device
        self.gsw = GSW()

        

    
    def get_est_loss(self, model, nepoch_per_unit, test_data, delta = 1/10):
        """
        Compute the final loss of the selected model.
        
        Args
        ----
        - model: the selected model  
        - nepoch_per_unit: size of a unit of resource
        - test_data: test data
        - delta: window size to estimate the finall loss (the proportion to self.max_iter)
        
        """
        dictionary = torch.load(model.dir + "/model.pth")
        units_trained = dictionary['epoch']
        model.load_state_dict(dictionary['model'])
        if model.model_type in ["DSWD", "DGSWD"]:
            model.tnet.load_state_dict(dictionary['tnet'])
            model.op_tnet.load_state_dict(dictionary['optnet'])
        elif model.model_type == "ASWD":
            model.phi.load_state_dict(dictionary['tnet'])
            model.phi_op.load_state_dict(dictionary['optnet'])
        
        R = int(self.max_iter/nepoch_per_unit)
        
        for parameter in model.tnet.parameters():
            parameter.requires_grad = False

        for parameter in model.phi.parameters():
            parameter.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, betas=(0.5, 0.999))
        optimizer.load_state_dict(dictionary['optimizer'])
        
        for parameter in model.tnet.parameters():
            parameter.requires_grad = True

        for parameter in model.phi.parameters():
            parameter.requires_grad = True
            
        dis = Discriminator(model.size, self.hidden_size).to(self.device)
        dis.load_state_dict(dictionary['dis'])
        
        disoptimizer = optim.Adam(dis.parameters(), lr=self.lr, betas=(0.5, 0.999))
        disoptimizer.load_state_dict(dictionary['disoptimizer'])
        
        
        lb = int(R - R*delta)
        print('Training the selected model...')
        self.run_model(dis, disoptimizer, model, optimizer, (lb-units_trained)*nepoch_per_unit, True)
        
        mmd_array = np.zeros(R-lb)
        print('Computing the esimated loss...')
        for i in range(lb+1, R+1):
            self.run_model(dis, disoptimizer, model, optimizer, nepoch_per_unit)  
            model.eval()
            fixednoise_wd = torch.randn((self.val_size, model.latent_size)).to("cuda")
            fake = model.decoder(fixednoise_wd)
            mmd_array[i-lb-1] = mmd.MMD(test_data.cpu().detach().numpy(), fake.cpu().detach().numpy(), 0.5)[0]
            
        return np.mean(mmd_array)
    
    
    
    
    
    def tune_adaptsh(self, T, B, nepoch_per_unit, test_data, test_data2, h, beta, delta = 1/10, alpha = 0.05, eta = 2):
        """
        Tune the hyperparameters using AdaptSH
        
        Args
        ----
        - T: list of models   
        - B: budget
        - nepoch_per_unit: size of a unit of resource
        - h: window size
        - beta: decay rate
        - delta: window size to estimate the finall loss (the proportion to self.max_iter)
        - alpha: significance level
        - eta: tuning parameter for SH
        
        Output: Name of the selected model
        
        """
#         best_configs = []
            
        optimizer = []
        for t in T:
            for parameter in t.tnet.parameters():
                parameter.requires_grad = False

            for parameter in t.phi.parameters():
                parameter.requires_grad = False
                    
            optimizer.append(optim.Adam(filter(lambda p: p.requires_grad, t.parameters()), lr=self.lr, betas=(0.5, 0.999)))

            for parameter in t.tnet.parameters():
                parameter.requires_grad = True

            for parameter in t.phi.parameters():
                parameter.requires_grad = True
            
        dis = []
        disoptimizer = []
        for i in range(len(T)):
            dis.append(Discriminator(T[i].size, self.hidden_size).to(self.device))
            disoptimizer.append(optim.Adam(dis[i].parameters(), lr=self.lr, betas=(0.5, 0.999)))      

        # initial number of models
        n = len(T)
        s = int(np.floor(np.log(n)/np.log(eta)))
        if int(B/s/n) <= 0:
            print('Budget too small!')
            import sys
            sys.exit()
        
        sum_r = 0
        R = int(self.max_iter/nepoch_per_unit)
        while s >= 1:
            r = int(B/s/n) 
            
            if r <= 0 or sum_r == R:
                break
            if sum_r + r <= R:
                sum_r += r
            else:
                r = R - sum_r
                sum_r = R
            
            
            print(
                "[*] training {} models for {} units of resources each".format(
                    n, r)
            )
            print("Cumulative # resources: ", sum_r)
            
            val_losses = []
            fake_samples = []
            fake_samples2 = []

            # train each of the remaining configs for r units
            for j, t in enumerate(T):  
                for i in range(r):
                    self.run_model(dis[j], disoptimizer[j], t, optimizer[j], nepoch_per_unit)  
                    if i >= r-h:
                        t.eval()
                        fixednoise_wd = torch.randn((self.val_size*2, t.latent_size)).to("cuda")
                        fake = t.decoder(fixednoise_wd)
                        np.savetxt(t.dir + '/sample_' + str(sum_r-r+1+i) + '.csv', fake.cpu().detach().numpy(), delimiter=',')
            
                t.eval()
                # save the intermediate model
                if sum_r <= int(R - R*delta):
                    if t.model_type in ["DSWD", "DGSWD"]:
                        save_dmodel(t, optimizer[j], dis[j], disoptimizer[j], t.tnet, t.op_tnet, sum_r, t.dir)
                    elif t.model_type == "ASWD":
                        save_dmodel(t, optimizer[j], dis[j], disoptimizer[j], t.phi, t.phi_op, sum_r, t.dir)
                    else:
                        save_dmodel(t, optimizer[j], dis[j], disoptimizer[j], None, None, sum_r, t.dir)
                
                
                fake_samples_t = []
                fake_samples_t2 = []

                for i in range(h):
                    if sum_r - i <= 0:
                        continue
                    fake = np.genfromtxt(t.dir + "/sample_" + str(sum_r-i) + ".csv", delimiter=',')
                    half = int(len(fake)//2)
                    fake_samples_t.append(fake[:half])
                    fake_samples_t2.append(fake[half:])

                fake_samples.append(fake_samples_t)
                fake_samples2.append(fake_samples_t2)
                
            # the remaining resources
            B -= n*r
            print('Remaining resources: ', B)     
            print()
            
            # pass the test and fake samples to the function that does the hypothesis testing
            insig_index = mmd.np_diff_mmd_test(test_data.cpu().detach().numpy(), test_data2.cpu().detach().numpy(), fake_samples, fake_samples2, 'fdr_by', alpha, beta, 10)

            T = [T[k] for k in insig_index]
            dis = [dis[k] for k in insig_index]
            disoptimizer = [disoptimizer[k] for k in insig_index] 
            optimizer = [optimizer[k] for k in insig_index] 
            n = len(T)
            s = int(np.floor(np.log(n)/np.log(eta)))

        best_model = T[0]

        return best_model.name 
    
    
    
    
    def tune_sh(self, T, B, nepoch_per_unit, test_data, delta = 1/10, alpha = 0.05, eta = 2):
        """
        Tune the hyperparameters using Successive Halving.
        
        Args
        ----
        - T: list of models   
        - B: budget
        - nepoch_per_unit: size of a unit of resource
        - delta: window size to estimate the finall loss (the proportion to self.max_iter)
        - alpha: significance level
        - eta: tuning parameter for SH
        
        Output: Name of the selected model
        
        """
#         best_configs = []
            
        optimizer = []
        for t in T:
            for parameter in t.tnet.parameters():
                parameter.requires_grad = False

            for parameter in t.phi.parameters():
                parameter.requires_grad = False
                    
            optimizer.append(optim.Adam(filter(lambda p: p.requires_grad, t.parameters()), lr=self.lr, betas=(0.5, 0.999)))

            for parameter in t.tnet.parameters():
                parameter.requires_grad = True

            for parameter in t.phi.parameters():
                parameter.requires_grad = True
            
        dis = []
        disoptimizer = []
        for i in range(len(T)):
            dis.append(Discriminator(T[i].size, self.hidden_size).to(self.device))
            disoptimizer.append(optim.Adam(dis[i].parameters(), lr=self.lr, betas=(0.5, 0.999)))      

        # initial number of models
        n = len(T)
        s = int(np.floor(np.log(n)/np.log(eta)))
        if int(B/s/n) <= 0:
            print('Budget too small!')
            import sys
            sys.exit()
        
        sum_r = 0
        R = int(self.max_iter/nepoch_per_unit)
        while s >= 1:
            r = int(B/s/n) 
            
            if r <= 0 or sum_r == R:
                break
            if sum_r + r <= R:
                sum_r += r
            else:
                r = R - sum_r
                sum_r = R
            
            
            print(
                "[*] training {} models for {} units of resources each".format(
                    n, r)
            )
            print("Cumulative # resources: ", sum_r)
            
            fake_samples = []

            # train each of the remaining configs for r units
            for j, t in enumerate(T): 
                self.run_model(dis[j], disoptimizer[j], t, optimizer[j], nepoch_per_unit*r)
            
                t.eval()
                # save the intermediate model
                if sum_r <= int(R - R*delta):
                    if t.model_type in ["DSWD", "DGSWD"]:
                        save_dmodel(t, optimizer[j], dis[j], disoptimizer[j], t.tnet, t.op_tnet, sum_r, t.dir)
                    elif t.model_type == "ASWD":
                        save_dmodel(t, optimizer[j], dis[j], disoptimizer[j], t.phi, t.phi_op, sum_r, t.dir)
                    else:
                        save_dmodel(t, optimizer[j], dis[j], disoptimizer[j], None, None, sum_r, t.dir)
                
                fixednoise_wd = torch.randn((self.val_size, t.latent_size)).to("cuda")
                fake = t.decoder(fixednoise_wd)
                fake_samples.append(fake.cpu().detach().numpy())
            
            val_losses = mmd.MMD_multi(test_data.cpu().detach().numpy(), fake_samples)
                
            # the remaining resources
            B -= n*r
            print('Remaining resources: ', B)     
            print()
            
            curr_best = T[np.argmin(val_losses)]
            sort_loss_idx = np.argsort(val_losses)
            n = int(len(T)//eta)
            T = [T[k] for k in sort_loss_idx[0:n]]
            dis = [dis[k] for k in sort_loss_idx[0:n]]
            disoptimizer = [disoptimizer[k] for k in sort_loss_idx[0:n]] 
            optimizer = [optimizer[k] for k in sort_loss_idx[0:n]] 
            
            s = int(np.floor(np.log(n)/np.log(eta)))
        # print(curr_best)
        best_model = curr_best

        return best_model.name 
    
    
    

    def run_model(self, dis, disoptimizer, model, optimizer, num_iters, show_progress = False):
        """
        Train a particular model for a
        given number of iterations.

        Args
        ----
        - model: the model to train.
        - num_iters: an int indicating the number of iterations
          to train the model for.

        """
        
        num_epochs = int(num_iters)
          
        if show_progress:
            for epoch in tqdm(range(num_epochs)):
                self._train_one_epoch(dis, disoptimizer, model, optimizer)
        else:
            for epoch in range(num_epochs):
                self._train_one_epoch(dis, disoptimizer, model, optimizer)
        
    
    
    def _train_one_epoch(self, dis, disoptimizer, model, optimizer):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.
        """
        model.train()

        train_loader = self.train_loader
        
        for i, data in enumerate(train_loader, start=0):
            if model.model_type == 'MSWD':
                loss = model.compute_loss_MSWD(dis, disoptimizer, data[0], torch.randn, self.gsw, model.max_iter)
            else: 
                loss = model.compute_loss(dis, disoptimizer, data[0], torch.randn)

            optimizer.zero_grad()

            if model.model_type in ["DSWD", "DGSWD"]:
                for name, parameter in model.named_parameters():
                    if name == 'tnet.net.0.weight':
                        parameter.requires_grad = False

            elif model.model_type == "ASWD":
                 for name, parameter in model.named_parameters():
                    if name == 'phi.net.0.weight':
                        parameter.requires_grad = False
            
            loss.backward()            
            optimizer.step()

            if model.model_type in ["DSWD", "DGSWD"]:
                for name, parameter in model.named_parameters():
                    if name == 'tnet.net.0.weight':
                        parameter.requires_grad = True

            elif model.model_type == "ASWD":
                 for name, parameter in model.named_parameters():
                    if name == 'phi.net.0.weight':
                        parameter.requires_grad = True

