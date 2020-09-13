import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import torch.distributions as D

class BaseModel(nn.Module):
    '''
    Base model defining loading and storing methods.
    '''
    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model

class GaussianModel(BaseModel):
    def __init__(self, n_classes=10, z_size=2):
        super(GaussianModel, self).__init__()
        self.loc = nn.Parameter(torch.randn(n_classes, z_size), requires_grad=True)
        self.cov_factor = nn.Parameter(torch.randn(n_classes, z_size, 1), requires_grad=True)
        self.cov_diag = nn.Parameter(torch.randn(n_classes, z_size), requires_grad=True)
        self.mix_weights = nn.Parameter(torch.ones(n_classes), requires_grad=False) # same weights
        self.n_classes = n_classes
        
    def forward(self, batch_size):
        mvn = D.LowRankMultivariateNormal(self.loc, self.cov_factor, torch.functional.F.elu(self.cov_diag) + 1)
        gmm = D.MixtureSameFamily(D.Categorical(self.mix_weights), mvn) # mix normal distributions
        return gmm.sample(torch.Size([batch_size]))
    
    def get_labels(self, loc):
        mvn = D.LowRankMultivariateNormal(self.loc, self.cov_factor, torch.functional.F.elu(self.cov_diag) + 1)
        loc = loc.repeat_interleave(self.n_classes, dim=0).reshape(loc.shape[0], self.n_classes, loc.shape[1])
        return F.softmax(torch.exp(mvn.log_prob(loc)), dim=1)
        

class Q_net(BaseModel):
    '''
    Encoder network.
    '''
    def __init__(self, input_size=784, hidden_size=1000, z_size=2, n_classes=10, dropout=0):
        super(Q_net, self).__init__()
        self.input_size = input_size
        self.dropout = dropout

        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        # batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn_y = nn.BatchNorm1d(n_classes)
        self.bn_z1 = nn.BatchNorm1d(z_size)
        self.bn_z2 = nn.BatchNorm1d(z_size)
        # gaussian encoding (z)
        self.lin3_class = nn.Linear(hidden_size, z_size)
        self.lin3_gauss = nn.Linear(hidden_size, z_size)
        # categorical label (y)
        # self.lin3_cat = nn.Linear(hidden_size, n_classes)

    def forward(self, x, labels=None):
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = F.relu(self.bn1(x))
        x = self.lin2(x)
        x = F.relu(self.bn2(x))
        z_class = self.bn_z1(self.lin3_class(x))
        z_gauss = self.bn_z2(self.lin3_gauss(x))
        return z_classs, z_gauss


class P_net(BaseModel):
    '''
    Decoder network.
    '''
    def __init__(self, input_size=784, hidden_size=1000, z_size=2, n_classes=10):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_size*2, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.lin1(x)
        #x = F.dropout(x, p=0, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0, training=self.training)
        x = self.lin3(x)
        return sigmoid(x)


class D_net(BaseModel):
    '''
    Discriminator network.
    '''
    def __init__(self, z_size=2, hidden_size=1000):
        super(D_net, self).__init__()
        self.lin1 = nn.Linear(z_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        #x = F.dropout(self.lin1(x), p=0, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        #x = F.dropout(self.lin2(x), p=0, training=self.training)
        x = F.relu(x)
        return sigmoid(self.lin3(x))
