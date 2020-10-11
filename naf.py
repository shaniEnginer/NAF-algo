class NAF(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed):
        super(NAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.mu = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
        
    def forward(self, input, action=None):
        """
        Forward pass of Normalized Advantage Function
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        mu = torch.tanh(self.mu(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)
        
        action_value = mu.unsqueeze(-1)
        
        # create lower-triangular matrix
        L = torch.zeros((input.shape[0], self.action_size, self.action_size)).to(device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        # exponentiate diagonal terms
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)
        
        Q = None
        if action is not None:
            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)
            Q = A + V
        
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)

        return action, Q, V
