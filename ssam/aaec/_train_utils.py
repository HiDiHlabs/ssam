import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

cuda = torch.cuda.is_available()


def add_noise(input, amount=0.3):
    '''
    Add random noise to input.
    '''
    noise = torch.randn(input.size()) * amount
    noisy_input = input + noise
    return noisy_input

def predict_labels(Q, X):
    Q.eval()

    latent_y = Q(X)[0]
    pred_labels = torch.argmax(latent_y, dim=1)
    return pred_labels

def get_categorial(label, n_classes=10):
    latent_y = np.eye(n_classes)[label].astype('float32')
    latent_y = torch.from_numpy(latent_y)
    return Variable(latent_y)

def sample_categorical(batch_size, n_classes=10, p=None):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     In case stated, a sampling probability given by p is used.
     return: torch.autograd.Variable with the sample
    '''
    #cat = np.random.randint(0, n_classes, batch_size)
    cat = np.random.choice(range(n_classes), size=batch_size, p=p)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)
    
def get_adversarial_categorial_weights(latent_y, batch_size, n_classes=10):
    '''
     Calculate the probabilties that will be used in the adversarial training 
     when training the categorial discriminator.
     The probabilities are the reverse probabilities of the abundance of each class
     in the output of the encoder (meaning less used classes, will be given higher weight).
     return: array, of probabilities
    '''
    pred_labels = torch.argmax(latent_y, dim=1)
    p = np.zeros((n_classes,))
    for label in pred_labels:
        p[label] += 1
    expected = batch_size / n_classes
    w = np.exp(expected - p)
    p_w = w / sum(w)
    return p_w

def classification_accuracy(Q, data_loader):
    correct = 0
    N = len(data_loader.dataset)

    for batch_idx, (X, target) in enumerate(data_loader):

        #X.resize_(data_loader.batch_size, Q.input_size)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        # encoding phase
        pred = predict_labels(Q, X)
        correct += pred.eq(target.data).cpu().sum()

    return 100. * correct / N

def unsupervised_classification_accuracy(Q, data_loader, n_classes=10):
    N = len(data_loader.dataset)

    pred_to_true = {}
    for _, (X, y) in enumerate(data_loader):

        X.resize_(data_loader.batch_size, Q.input_size)

        X, y = Variable(X), Variable(y)
        if cuda:
            X, y = X.cuda(), y.cuda()

        y_pred = predict_labels(Q, X)

        for y_true, y_hat in zip(y, y_pred):
            pred_to_true.setdefault(y_hat.item(), {})
            pred_to_true[y_hat.item()].setdefault(y_true.item(), 0)
            pred_to_true[y_hat.item()][y_true.item()] += 1

    correct = 0
    for y_hat in range(n_classes):
        try:
            best_matching_label = max(pred_to_true[y_hat], key=pred_to_true[y_hat].get)
            correct += pred_to_true[y_hat][best_matching_label]
        except:
            print("\nlabel %s in never predicted" % y_hat)

    return 100. * correct / N

def get_unsupervised_boosting_weights(Q, P, train_unlabeled_loader, valid_loader):
    #### Get sample weights (boosting)
    batch_size = train_unlabeled_loader.batch_size
    
    weights = torch.Tensor()
    if cuda:
        weights = weights.cuda()
        
    for batch_num, (X, target) in enumerate(train_unlabeled_loader):

        X.resize_(batch_size, Q.input_size)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        latent_vec = torch.cat(Q(X), 1)
        X_rec = P(latent_vec)
        
        for x, x_rec, y_true in zip(X, X_rec, target):
            # Reconstruction loss
            loss = F.binary_cross_entropy(x_rec, x)
            weights = torch.cat((weights, torch.unsqueeze(loss, dim=0)))

    weights = (weights - torch.min(weights))/ (torch.max(weights) - torch.min(weights))
        
    ## for validation
    weights_per_label = {}
    for batch_num, (X, target) in enumerate(valid_loader):

        X.resize_(batch_size, Q.input_size)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        latent_vec = torch.cat(Q(X), 1)
        X_rec = P(latent_vec)
        
        for x, x_rec, y_true in zip(X, X_rec, target):
            # Reconstruction loss
            loss = F.binary_cross_entropy(x_rec, x)

            weights_per_label.setdefault(y_true.item(), 0)
            weights_per_label[y_true.item()] += loss.item()
            
    highest_weight_label = max(weights_per_label, key=weights_per_label.get)
    print("\nhighest label weights is the digit {}".format(highest_weight_label))
    ######
    
    return weights
    
def zero_grad_all(*models):
    [m.zero_grad() for m in models]

def train_all(*models):
    [m.train() for m in models]

def eval_all(*models):
    [m.eval() for m in models]

def report_loss(epoch, all_losses, descriptions, output_dir=None):
    '''
    Print loss.
    '''
    base_loss_report = 'Epoch-{}; '.format(epoch)

    for loss, desc in zip(all_losses, descriptions):
        if loss is None:
            base_loss_report += '{}: N/A; '.format(desc)
        else:
            base_loss_report += '{}: {:.4}; '.format(desc, loss.item())

    if output_dir:
        with open(os.path.join(output_dir, 'loss_report.txt'), 'a') as f_report:
            f_report.write(base_loss_report)

    print(base_loss_report)

def report_progress(percent, barLen=20):
    sys.stdout.write("\rcurrent epoch:: ")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()
