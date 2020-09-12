import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


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
        self.bn_z = nn.BatchNorm1d(z_size)
        # gaussian encoding (z)
        self.lin3_gauss = nn.Linear(hidden_size, z_size)
        # categorical label (y)
        self.lin3_cat = nn.Linear(hidden_size, n_classes)

    def forward(self, x, labels=None):
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = F.relu(self.bn1(x))
        x = self.lin2(x)
        x = F.relu(self.bn2(x))
        z_gauss = self.bn_z(self.lin3_gauss(x))
        y_cat = F.softmax(self.bn_y(self.lin3_cat(x)), dim=1)
        return y_cat, z_gauss


class P_net(BaseModel):
    '''
    Decoder network.
    '''
    def __init__(self, input_size=784, hidden_size=1000, z_size=2, n_classes=10):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_size + n_classes, hidden_size)
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


class D_net_cat(BaseModel):
    '''
    Categorical descriminator network.
    '''
    def __init__(self, n_classes=10, hidden_size=1000):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(n_classes, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return sigmoid(x)


class D_net_gauss(BaseModel):
    '''
    Gaussian descriminator network.
    '''
    def __init__(self, z_size=2, hidden_size=1000):
        super(D_net_gauss, self).__init__()
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
