import torch
cuda = torch.cuda.is_available()

import itertools
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ._model import Q_net, P_net, D_net_cat, D_net_gauss
from ._train_utils import *


def _train_epoch(
    models, optimizers, train_unlabeled_loader, n_classes, z_dim, config_dict):
    '''
    Train procedure for one epoch.
    '''
    epsilon = np.finfo(float).eps
    params = config_dict['training']

    # load models and optimizers
    P, Q, D_cat, D_gauss, P_mode_decoder = models
    auto_encoder_optim, G_optim, D_optim, info_optim, mode_optim, disentanglement_optim = optimizers

    # Set the networks in train mode (apply dropout when needed)
    train_all(P, Q, D_cat, D_gauss, P_mode_decoder)

    batch_size = train_unlabeled_loader.batch_size
    n_batches = len(train_unlabeled_loader)

    # Loop through the unlabeled dataset
    for batch_num, (X, target) in enumerate(train_unlabeled_loader):
        X_noisy = add_noise(X)

        X, X_noisy, target = Variable(X), Variable(X_noisy), Variable(target)
        if cuda:
            X, X_noisy, target = X.cuda(), X_noisy.cuda(), target.cuda()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Reconstruction phase
        #######################
        latent_vec = torch.cat(Q(X_noisy), 1)
        X_rec = P(latent_vec)

        recon_loss = F.mse_loss(X_rec + epsilon, X + epsilon)
        
        recon_loss.backward()
        auto_encoder_optim.step()
        
        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Info phase
        #######################
        continuous_loss = torch.nn.MSELoss()

        latent_y, latent_z = Q(X)
        latent_vec = torch.cat((latent_y, latent_z), 1)
        X_rec = P(latent_vec)

        latent_y_rec, latent_z_rec = Q(X_rec)

        cat_info_loss = F.binary_cross_entropy(latent_y_rec, latent_y.detach())
        gauss_info_loss = continuous_loss(latent_z_rec, latent_z.detach())

        mutual_info_loss = 1.0 * cat_info_loss + 0.1 * gauss_info_loss

        if params['use_mutual_info']:
            mutual_info_loss.backward()
            info_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Mode reconstruction phase
        #######################
        latent_y, latent_z = Q(X_noisy)
        X_mode_rec = P_mode_decoder(latent_y)

        mode_recon_loss = F.binary_cross_entropy(X_mode_rec + epsilon, X + epsilon)

        if params['use_mode_decoder']:
            mode_recon_loss.backward()
            mode_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Mode disentanglement phase
        #######################
        mode_disentanglement_loss = 0
            
        latent_z_all_zeros = Variable(torch.zeros(z_dim))
        
        for label_A in range(n_classes):
            latent_y_A = get_categorial(label_A, n_classes=n_classes)
        
            latent_vec_A = torch.cat((latent_y_A, latent_z_all_zeros), 0)
            if cuda:
                latent_vec_A = latent_vec_A.cuda()
            X_mode_rec_A = P(latent_vec_A)
        
            for label_B in range(label_A + 1, n_classes):
                latent_y_B = get_categorial(label_B, n_classes=n_classes)
        
                latent_vec_B = torch.cat((latent_y_B, latent_z_all_zeros), 0)
                if cuda:
                    latent_vec_B = latent_vec_B.cuda()
                X_mode_rec_B = P(latent_vec_B)
        
                mode_disentanglement_loss += -F.binary_cross_entropy(X_mode_rec_A + epsilon, X_mode_rec_B.detach() + epsilon)
        
        mode_disentanglement_loss /= (n_classes * (n_classes - 1) / 2)
        
        if params['use_disentanglement']:
            mode_disentanglement_loss.backward()
            disentanglement_optim.step()
        
        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Generator phase
        #######################
        Q.train()
        z_fake_cat, z_fake_gauss = Q(X)

        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = - torch.mean(torch.log(D_fake_cat + epsilon)) - torch.mean(torch.log(D_fake_gauss + epsilon))
        G_loss += z_fake_gauss.norm() * params['lambda_z_l2_regularization']
        
        G_loss.backward()
        G_optim.step()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        #######################
        # Discriminator phase
        #######################
        Q.eval()
        z_fake_cat, z_fake_gauss = Q(X)

        p_cat = None
        if params['use_adversarial_categorial_weights']:
            p_cat = get_adversarial_categorial_weights(z_fake_cat, batch_size, n_classes=n_classes)
        
        z_real_cat = sample_categorical(batch_size, n_classes=n_classes, p=p_cat)
        z_real_gauss = Variable(torch.randn(batch_size, z_dim))
        if cuda:
            z_real_cat = z_real_cat.cuda()
            z_real_gauss = z_real_gauss.cuda()

        D_real_cat = D_cat(z_real_cat)
        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_cat = D_cat(z_fake_cat)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss_cat = - torch.mean(torch.log(D_real_cat + epsilon) + torch.log(1 - D_fake_cat + epsilon))
        D_loss_gauss = - torch.mean(torch.log(D_real_gauss + epsilon) + torch.log(1 - D_fake_gauss + epsilon))

        D_loss = D_loss_cat + D_loss_gauss

        D_loss.backward()
        D_optim.step()

        Q.train()

        # Init gradients
        zero_grad_all(P, Q, D_cat, D_gauss, P_mode_decoder)

        # report progress
        report_progress(float(batch_num) / n_batches)
        
    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, mode_recon_loss, mutual_info_loss, mode_disentanglement_loss


def _get_optimizers(models, config_dict, decay=1.0):
    '''
    Set and return all relevant optimizers needed for the training process.
    '''
    P, Q, D_cat, D_gauss, P_mode_decoder = models

    # Set learning rates
    learning_rates = config_dict['learning_rates']

    auto_encoder_lr = learning_rates['auto_encoder_lr'] * decay
    generator_lr = learning_rates['generator_lr'] * decay
    discriminator_lr = learning_rates['discriminator_lr'] * decay
    info_lr = learning_rates['info_lr'] * decay
    mode_lr = learning_rates['mode_lr'] * decay
    disentanglement_lr = learning_rates['disentanglement_lr'] * decay

    # Set optimizators
    auto_encoder_optim = optim.Adam(itertools.chain(Q.parameters(), P.parameters()), lr=auto_encoder_lr)

    G_optim = optim.Adam(Q.parameters(), lr=generator_lr)
    D_optim = optim.Adam(itertools.chain(D_gauss.parameters(), D_cat.parameters()), lr=discriminator_lr)

    info_optim = optim.Adam(itertools.chain(Q.parameters(), P.parameters()), lr=info_lr)
    mode_optim = optim.Adam(itertools.chain(Q.parameters(), P_mode_decoder.parameters()), lr=mode_lr)
    disentanglement_optim = optim.Adam(P.parameters(), lr=disentanglement_lr)
    
    if not config_dict['training']['use_adam_optimization']:
        auto_encoder_optim = optim.SGD(itertools.chain(Q.parameters(), P.parameters()), lr=0.01 * decay, momentum=0.9)
        G_optim = optim.SGD(Q.parameters(), lr=0.1 * decay, momentum=0.1)
        D_optim = optim.SGD(itertools.chain(D_gauss.parameters(), D_cat.parameters()), lr=0.1 * decay, momentum=0.1)

    optimizers = auto_encoder_optim, G_optim, D_optim, info_optim, mode_optim, disentanglement_optim

    return optimizers


def _get_models(n_classes, n_features, z_dim, config_dict):
    '''
    Set and return all sub-modules that comprise the full model.
    '''
    
    model_params = config_dict['model']

    Q = Q_net(
        z_size=z_dim,
        n_classes=n_classes,
        input_size=n_features,
        hidden_size=model_params['hidden_size'],
        dropout=model_params['encoder_dropout'])
    P = P_net(z_size=z_dim, n_classes=n_classes, input_size=n_features, hidden_size=model_params['hidden_size'])
    D_cat = D_net_cat(n_classes=n_classes, hidden_size=model_params['hidden_size'])
    D_gauss = D_net_gauss(z_size=z_dim, hidden_size=model_params['hidden_size'])

    # Introducing the new Mode-decoder (it only gets the mode latent y)
    P_mode_decoder = P_net(z_size=0, n_classes=n_classes, input_size=n_features, hidden_size=model_params['hidden_size'])

    if cuda:
        Q = Q.cuda()
        P = P.cuda()
        D_gauss = D_gauss.cuda()
        D_cat = D_cat.cuda()
        P_mode_decoder = P_mode_decoder.cuda()

    models = P, Q, D_cat, D_gauss, P_mode_decoder
    return models

def train(train_unlabeled_loader, epochs, n_classes, n_features, z_dim, output_dir, config_dict, verbose):
    '''
    Train the full model.
    '''
    learning_curve = []

    models = _get_models(n_classes, n_features, z_dim, config_dict)
    optimizers = _get_optimizers(models, config_dict)
    P, Q, D_cat, D_gauss, P_mode_decoder = models

    for epoch in range(epochs):
        if epoch == 50: # learning rate decay
            optimizers = _get_optimizers(models, config_dict, decay=0.1)

        all_losses = _train_epoch(
            models,
            optimizers,
            train_unlabeled_loader,
            n_classes,
            z_dim,
            config_dict)

        learning_curve.append(all_losses)

        if verbose:
            report_loss(
                epoch+1,
                all_losses,
                descriptions=['D_loss_cat', 'D_loss_gauss', 'G_loss', 'recon_loss', 'mode_recon_loss', 'mutual_info_loss', 'mode_disentanglement_loss'],
                output_dir=output_dir)

    return Q, P, P_mode_decoder, learning_curve
