import numpy as np
import torch
from torch import optim
from TransformNet import TransformNet



def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, p=2, device="cuda"):
    dim = second_samples.size(1)
#     print(num_projections)
    projections = rand_projections(dim, num_projections).to(device)
#     print(projections.shape)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
            torch.sort(first_projections.transpose(0, 1), dim=1)[0]
            - torch.sort(second_projections.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=1)
    return torch.pow(wasserstein_distance.mean(), 1.0 / p)



def max_sliced_wasserstein_distance(first_samples, second_samples, p=2, max_iter=100, device="cuda"):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=1e-4)
    for _ in range(max_iter):
        encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
        distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (torch.sort(encoded_projections)[0] - torch.sort(distribution_projections)[0])
        )
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))
        l = -wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    
    encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
    distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (torch.sort(encoded_projections)[0] - torch.sort(distribution_projections)[0])
    )
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=1)
    return torch.pow(wasserstein_distance.mean(), 1.0 / p)



def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix_slow(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix_slow(x2, theta * r))
    wasserstein_distance = torch.abs(
        (torch.sort(cost_matrix_1.transpose(0, 1), dim=1)[0] - torch.sort(cost_matrix_2.transpose(0, 1), dim=1)[0])
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)


def generalized_sliced_wasserstein_distance(
    first_samples, second_samples, g_fuction, r=1, num_projections=1000, p=2, device="cuda"
):
    embedding_dim = first_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    return g_fuction(first_samples, second_samples, projections, r, p)



def max_generalized_sliced_wasserstein_distance(
    first_samples, second_samples, g_function, r, p=2, max_iter=100, device="cuda"
):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=1e-4)
    for _ in range(max_iter):
        wasserstein_distance = g_function(first_samples, second_samples, theta, r, p)
        l = -wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    wasserstein_distance = g_function(first_samples, second_samples, theta, r, p)
    return wasserstein_distance


def distributional_generalized_sliced_wasserstein_distance(
    first_samples, second_samples, num_projections, f, f_op, g_function, r, p=2, max_iter=10, lam=1, device="cuda"
):
    embedding_dim = first_samples.size(1)
#     print(embedding_dim)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    for _ in range(max_iter):
        projections = f(pro)
        reg = lam * cosine_distance_torch(projections, projections)
        wasserstein_distance = g_function(first_samples, second_samples, projections, r, p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    projections = f(pro)
    wasserstein_distance = g_function(first_samples, second_samples, projections, r, p)
    return wasserstein_distance



def augmented_sliced_wassersten_distance(first_samples, second_samples, num_projections, phi,
                                                       phi_op, p=2, max_iter=10, lam=0.5, device='cuda'):
    embedding_dim = first_samples.size(1)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        first_samples_transform = phi(first_samples_detach)
        second_samples_transform = phi(second_samples_detach)
        reg = lam * (torch.norm(first_samples_transform, p=2, dim=1) + torch.norm(second_samples_transform, p=2,
                                                                                  dim=1)).mean()
        projections = rand_projections(first_samples_transform.shape[-1], num_projections).to(device)
        encoded_projections = first_samples_transform.matmul(projections.transpose(0, 1))
        distribution_projections = (second_samples_transform.matmul(projections.transpose(0, 1)))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                          torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=1)
        wasserstein_distance = torch.pow(wasserstein_distance.mean(), 1. / p)
        loss = reg - wasserstein_distance
        phi_op.zero_grad()
        loss.backward(retain_graph=True)
        phi_op.step()
    
    first_samples_transform = phi(first_samples)
    second_samples_transform = phi(second_samples)
    projections = rand_projections(first_samples_transform.shape[-1], num_projections).to(device)
    encoded_projections = first_samples_transform.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples_transform.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=1)
    wasserstein_distance = torch.pow(wasserstein_distance.mean(), 1. / p)
    return wasserstein_distance


def distributional_sliced_wasserstein_distance(
    first_samples, second_samples, num_projections, f, f_op, p=2, max_iter=10, lam=1, device="cuda"
):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (
                torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
            )
        )
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
            torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
            - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance



def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))



def cost_matrix_slow(x, y):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)



def save_dmodel(model, optimizer, dis, disoptimizer, tnet, optnet, epoch, folder):
    
    dictionary = {}
    dictionary["epoch"] = epoch
    dictionary["model"] = model.state_dict()
    if optimizer:
        dictionary["optimizer"] = optimizer.state_dict()
    if disoptimizer:
        dictionary["dis"] = dis.state_dict()
        dictionary["disoptimizer"] = disoptimizer.state_dict()
    else:
        dictionary["dis"] = None
        dictionary["disoptimizer"] = None
    if tnet:
        dictionary["tnet"] = tnet.state_dict()
        dictionary["optnet"] = optnet.state_dict()
    else:
        dictionary["tnet"] = None
        dictionary["optnet"] = None
    
    torch.save(dictionary, folder + "/model" + ".pth")

