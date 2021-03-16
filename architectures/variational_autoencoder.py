import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.mu = nn.Linear(hidden_size * 2, latent_size)
        self.log_var = nn.Linear(hidden_size * 2, latent_size)


        self.dec = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(latent_size, hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Linear(hidden_size * 2, input_size)
        )

    def encode(self, x):
        x = nn.Dropout(0.1)(x)
        x = self.fc1(x)
        x = nn.Dropout(0.1)(x)
        x = self.fc2(x)
        x = nn.Dropout(0.1)(x)
        return self.mu(x), self.log_var(x)

    def decode(self, z):
        return self.dec(z)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

    def sample(self, timesteps, x):
        self.eval()
        out = []
        for i in range(timesteps):
            x, mu, logvar = self.forward(x)
            out.append(x)
        return torch.cat(out, dim=0)