import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gswnn import GSW_NN
from simpleGAN import simpleAutoencoder
from torch import optim
from TransformNet import TransformNet, Mapping
from utils import circular_function
from generate import *
import joblib

from tuning import Tuning


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def main():
    # train args
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning using Adaptive Resource Allocation")
    parser.add_argument("--datadir", default="./data", help="path to dataset")
    parser.add_argument("--outdir", default="./result", help="directory to output")
    parser.add_argument(
        "--epochs", type=int, default=5000, metavar="N", help="number of maximum epochs to train (default: 5000)"
    )
    
    parser.add_argument("--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.0005)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        metavar="N",
        help="number of dataloader workers if device is CPU (default: 16)",
    )
    parser.add_argument("--seed", type=int, default=16, metavar="S", help="random seed (default: 16)")
    parser.add_argument("--latent-size", type=int, default=2, help="Latent size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size")
    parser.add_argument("--dataset", type=str, default="SWISSROLL", help="(SWISSROLL|MOON)")
    parser.add_argument("--train-size", type=int, default=1000, help="training sample size")
    parser.add_argument("--test-size", type=int, default=500, help="test sample size")
    parser.add_argument("--h", type=int, default=6, help="window size (default: 6)")
    parser.add_argument("--beta", type=float, default=0.9, help="decay rate (default: 0.9)")
    parser.add_argument("--B", type=int, default=1000, help="budget (default: 1000)")
    parser.add_argument("--unit-size", type=int, default=10, help="size of one unit of resources (default: 10)")
    parser.add_argument("--algorithm", type=str, default="AdaptSH", help="(AdaptSH|SH)")

    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    latent_size = args.latent_size
    dataset = args.dataset
    assert dataset in ["SWISSROLL", "MOON"]
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    datadir = os.path.join(args.datadir, args.dataset)
    outdir = os.path.join(args.outdir, args.dataset)
    sampledir = os.path.join(outdir, 'intermediate_samples')
    if not (os.path.isdir(datadir)):
        os.makedirs(datadir)
    if not (os.path.isdir(outdir)):
        os.makedirs(outdir)
    if not (os.path.isdir(sampledir)):
        os.makedirs(sampledir)
        
    
    if dataset == "SWISSROLL": 
        generator = generate_swiss_roll_3d              
        dim = 3

    elif dataset == "MOON":
        generator = generate_moons
        dim = 2
    
    
    try:
        training_data = joblib.load(datadir + '/training_data.joblib')
    except:
        print('Training data not exst, creating new training data')
        training_data = generator(args.train_size).to(device)
        joblib.dump(training_data, datadir + '/training_data.joblib')
        
    try:
        test_data = joblib.load(datadir + '/test_data.joblib')
    except:
        print('Test data1 not exst, creating new test data1')
        test_data = generator(args.test_size).to(device)
        joblib.dump(test_data, datadir + '/test_data.joblib')
        
    try:
        test_data2 = joblib.load(datadir + '/test_data2.joblib')
    except:
        print('Test data2 not exst, creating new test data2')
        test_data2 = generator(args.test_size).to(device)
        joblib.dump(test_data2, datadir + '/test_data2.joblib')
    
    
    hyper_tuning = Tuning(args, training_data, device)    

# 30 models
    models = {
        'ASWD': {'lam': [1, 5], 'max_iter': [10, 20], 'lr': [0.005, 0.0005]},              
        'MSWD': {'max_iter': [10, 50]},
        'SWD': {'num_proj': [10, 1000]},        
        'GSWD': {'r': [2, 5], 'num_proj': [10, 1000]},
        'MGSWD': {'r': [2, 5], 'max_iter': [10, 50]},
        'MGSWNN': {'max_iter': [10, 50]},
        'DSWD': {'lam': [1, 5], 'max_iter': [10, 20], 'lr': [0.005, 0.0005]}      
    }
    
    
    
    for i in range(1):
        T = []

        for model_type, params in models.items():
            if model_type == 'SWD':
                num_proj_list = params['num_proj']
                for num_proj in num_proj_list:
                    model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                    model.num_projection = num_proj
                    model.dir = os.path.join(sampledir, model_type + "_n" + str(num_proj))
                    model.name = model_type + "_n" + str(num_proj)
                    T.append(model)                
            elif model_type == 'MSWD':
                iter_list = params['max_iter']
                for max_iter in iter_list:
                    model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                    model.max_iter = max_iter
                    model.dir = os.path.join(sampledir, model_type + "_miter" + str(max_iter))
                    model.name = model_type + "_miter" + str(max_iter)
                    T.append(model)
            elif model_type == 'GSWD':
                r_list = params['r']
                num_proj_list = params['num_proj']
                for r in r_list:
                    for num_proj in num_proj_list:
                        model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                        model.g_function = circular_function
                        model.r = r
                        model.num_projection = num_proj
                        model.dir = os.path.join(sampledir, model_type + "_r" + str(r) + "_nproj" + str(num_proj))
                        model.name = model_type + "_r" + str(r) + "_nproj" + str(num_proj)
                        T.append(model)
            elif model_type == 'MGSWD':
                r_list = params['r']
                iter_list = params['max_iter']
                for r in r_list:
                    for max_iter in iter_list:
                        model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                        model.g_function = circular_function
                        model.r = r
                        model.max_iter = max_iter
                        model.dir = os.path.join(sampledir, model_type + "_r" + str(r) + "_niter" + str(max_iter))
                        model.name = model_type + "_r" + str(r) + "_niter" + str(max_iter)
                        T.append(model)
            elif model_type == 'MGSWNN':
                iter_list = params['max_iter']
                for max_iter in iter_list:
                    model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                    model.max_iter = max_iter
                    model.gsw = GSW_NN(din=args.hidden_size, nofprojections=1, model_depth=3, num_filters=3, use_cuda=True)
                    model.dir = os.path.join(sampledir, model_type + "_niter" + str(max_iter))
                    model.name = model_type + "_niter" + str(max_iter)
                    T.append(model)
            elif model_type == 'DSWD':
                lam_list = params['lam']
                iter_list = params['max_iter']
                lr_list = params['lr']
                for lam in lam_list:
                    for max_iter in iter_list:
                        for lr in lr_list:
                            model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                            model.lam = lam
                            model.max_iter = max_iter                    
                            # model.tnet = TransformNet(args.hidden_size).to(device)
                            model.op_tnet = optim.Adam(model.tnet.parameters(), lr=lr, betas=(0.5, 0.999))
                            model.dir = os.path.join(
                                sampledir, model_type + "_niter" + str(max_iter) + "_lr" + str(lr) + "_lam" + str(lam)
                            )
                            model.name = model_type + "_niter" + str(max_iter) + "_lr" + str(lr) + "_lam" + str(lam)
                            T.append(model)

            elif model_type == 'DGSWD':
                lam_list = params['lam']
                r_list = params['r']
                lr_list = params['lr']
                for lam in lam_list:
                    for r in r_list:
                        for lr in lr_list:
                            model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                            model.g_function = circular_function
                            model.lam = lam
                            model.r = r                   
                            # model.tnet = TransformNet(args.hidden_size).to(device)
                            model.op_tnet = optim.Adam(model.tnet.parameters(), lr=lr, betas=(0.5, 0.999))
                            model.dir = os.path.join(
                                sampledir, model_type + "_r" + str(r) + "_lr" + str(lr) + "_lam" + str(lam)
                            )
                            model.name = model_type + "_r" + str(r) + "_lr" + str(lr) + "_lam" + str(lam)
                            T.append(model)                
            elif model_type == 'ASWD':
                lam_list = params['lam']
                iter_list = params['max_iter']
                lr_list = params['lr']
                for lam in lam_list:
                    for max_iter in iter_list:
                        for lr in lr_list:
                            model = simpleAutoencoder(size=dim, latent_size=latent_size, hidden_size=args.hidden_size, device=device, model_type = model_type).to(device)
                            model.lam = lam
                            model.max_iter = max_iter
                            # model.phi = Mapping(hidden_size).to(device)
                            model.phi_op = optim.Adam(model.phi.parameters(), lr=lr, betas=(0.5, 0.999))
                            model.dir = os.path.join(
                                sampledir, model_type + "_niter" + str(max_iter) + "_lr" + str(lr) + "_lam" + str(lam)
                            )
                            model.name = model_type + "_niter" + str(max_iter) + "_lr" + str(lr) + "_lam" + str(lam)
                            T.append(model)

        print(
            "narms {}\nnepochs {}\nAdam lr {} \nusing device {}\n".format(
                len(T), args.epochs, args.lr, device.type
            )
        )
        
        for model in T:
            if not (os.path.isdir(model.dir)):
                os.makedirs(model.dir)

        
        
        if args.algorithm == 'SH':
            selected_model = hyper_tuning.tune_sh(T, args.B, args.unit_size, test_data, 1/10)
        else:
            selected_model = hyper_tuning.tune_adaptsh(T, args.B, args.unit_size, test_data, test_data2, args.h, args.beta, 1/10)
        print('Selected Configuration:')
        print(selected_model)
        
        the_model = T[np.where([i.name == selected_model for i in T])[0][0]]
        est_loss = hyper_tuning.get_est_loss(the_model, args.unit_size, test_data, 1/10)
        print(est_loss)



if __name__ == "__main__":
    main()
