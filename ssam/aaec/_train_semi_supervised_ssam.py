import torch
import itertools
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ._model import Q_net, P_net, D_net_cat, D_net_gauss
from ._train_utils import *

cuda = torch.cuda.is_available()


def _train_epoch(
    models, optimizers, train_labeled_loader, train_unlabeled_loader, n_classes, z_dim, config_dict):
    '''
    Train procedure for one epoch.
    '''
    epsilon = np.finfo(float).eps

    # load models and optimizers
    P, Q, D_cat, D_gauss = models
    auto_encoder_optim, G_optim, D_optim, classifier_optim = optimizers

    # Set the networks in train mode (apply dropout when needed)
    train_all(P, Q, D_cat, D_gauss)

    batch_size = train_labeled_loader.batch_size

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    for (X_l, target_l), (X_u, target_u) in zip(train_labeled_loader, train_unlabeled_loader):

        for X, target in [(X_u, target_u), (X_l, target_l)]:
            if target[0] == -1:
                labeled = False
            else:
                labeled = True

            #X.resize_(batch_size, Q.input_size)

            X_noisy = add_noise(X)

            X, X_noisy, target = Variable(X), Variable(X_noisy), Variable(target)
            if cuda:
                X, X_noisy, target = X.cuda(), X_noisy.cuda(), target.cuda()


            # Init gradients
            zero_grad_all(P, Q, D_cat, D_gauss)

            if not labeled:
                #######################
                # Reconstruction phase
                #######################
                latent_vec = torch.cat(Q(X_noisy), 1)
                X_rec = P(latent_vec)

                recon_loss = F.mse_loss(X_rec + epsilon, X + epsilon)

                recon_loss.backward()
                auto_encoder_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

                #######################
                # Discriminator phase
                #######################
                Q.eval()
                z_real_cat = sample_categorical(batch_size, n_classes=n_classes)
                z_real_gauss = Variable(torch.randn(batch_size, z_dim))
                if cuda:
                    z_real_cat = z_real_cat.cuda()
                    z_real_gauss = z_real_gauss.cuda()

                z_fake_cat, z_fake_gauss = Q(X)

                D_real_cat = D_cat(z_real_cat)
                D_real_gauss = D_gauss(z_real_gauss)
                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                D_loss_cat = - torch.mean(torch.log(D_real_cat + epsilon) + torch.log(1 - D_fake_cat + epsilon))
                D_loss_gauss = - torch.mean(torch.log(D_real_gauss + epsilon) + torch.log(1 - D_fake_gauss + epsilon))

                D_loss = D_loss_cat + D_loss_gauss

                D_loss.backward()
                D_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

                #######################
                # Generator phase
                #######################
                Q.train()
                z_fake_cat, z_fake_gauss = Q(X)

                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                G_loss = - torch.mean(torch.log(D_fake_cat + epsilon)) - torch.mean(torch.log(D_fake_gauss + epsilon))

                G_loss.backward()
                G_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

            #######################
            # Semi-supervised phase
            #######################
            if labeled:
                pred, _ = Q(X)
                class_loss = F.cross_entropy(pred, target)
                class_loss.backward()
                classifier_optim.step()

                # Init gradients
                zero_grad_all(P, Q, D_cat, D_gauss)

    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss


def _get_optimizers(models, config_dict, decay=1.0):
    '''
    Set and return all relevant optimizers needed for the training process.
    '''
    P, Q, D_cat, D_gauss = models

    # Set learning rates
    learning_rates = config_dict['learning_rates']

    auto_encoder_lr = learning_rates['auto_encoder_lr'] * decay
    generator_lr = learning_rates['generator_lr'] * decay
    discriminator_lr = learning_rates['discriminator_lr'] * decay
    classifier_lr = learning_rates['classifier_lr'] * decay

    # Set optimizators
    auto_encoder_optim = optim.Adam(itertools.chain(Q.parameters(), P.parameters()), lr=auto_encoder_lr)

    G_optim = optim.Adam(Q.parameters(), lr=generator_lr)
    D_optim = optim.Adam(itertools.chain(D_gauss.parameters(), D_cat.parameters()), lr=discriminator_lr)

    classifier_optim = optim.Adam(Q.parameters(), lr=classifier_lr)

    optimizers = auto_encoder_optim, G_optim, D_optim, classifier_optim

    return optimizers


def _get_models(n_classes, n_features, z_dim, config_dict):
    '''
    Set and return all sub-modules that comprise the full model.
    '''
    model_params = config_dict['model']

    Q = Q_net(z_size=z_dim, n_classes=n_classes, input_size=n_features, hidden_size=model_params['hidden_size'])
    P = P_net(z_size=z_dim, n_classes=n_classes, input_size=n_features, hidden_size=model_params['hidden_size'])
    D_cat = D_net_cat(n_classes=n_classes, hidden_size=model_params['hidden_size'])
    D_gauss = D_net_gauss(z_size=z_dim, hidden_size=model_params['hidden_size'])

    if cuda:
        Q = Q.cuda()
        P = P.cuda()
        D_gauss = D_gauss.cuda()
        D_cat = D_cat.cuda()

    models = P, Q, D_cat, D_gauss
    return models


def train(train_labeled_loader, train_unlabeled_loader, valid_loader, epochs, n_classes, n_features, z_dim, output_dir, config_dict):
    '''
    Train the full model.
    '''
    learning_curve = []

    models = _get_models(n_classes, n_features, z_dim, config_dict)
    optimizers = _get_optimizers(models, config_dict)
    P, Q, D_cat, D_gauss = models

    for epoch in range(epochs):
        if epoch == 50: # learning rate decay
            optimizers = _get_optimizers(models, config_dict, decay=0.1)

        all_losses = _train_epoch(
            models,
            optimizers,
            train_labeled_loader,
            train_unlabeled_loader,
            n_classes,
            z_dim,
            config_dict)

        learning_curve.append(all_losses)

        report_loss(
            epoch+1,
            all_losses,
            descriptions=['D_loss_cat', 'D_loss_gauss', 'G_loss', 'recon_loss', 'class_loss'],
            output_dir=output_dir)
        
        if epoch % 10 == 9:
            val_acc = classification_accuracy(Q, valid_loader)
            print('Validation accuracy: {} %'.format(val_acc))

    return Q, P, learning_curve
