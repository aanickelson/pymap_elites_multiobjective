import torch
import torch.nn as nn
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, nstate=4, nbh=2):
        super(Autoencoder, self).__init__()

        self.S1 = 60
        self.S2 = 30
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Linear(nstate, self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            # nn.Sigmoid(),
            nn.Linear(self.S1, self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            # nn.Sigmoid(),
            nn.Linear(self.S2, nbh),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(nbh, self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            # nn.Sigmoid(),
            nn.Linear(self.S2, self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            # nn.Sigmoid(),
            nn.Linear(self.S1, nstate)
        )

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        lr = 1e-2
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def feed(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        pred = self.encode(x)
        return pred.detach().numpy()

    def test(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        pred = self.encode(x)
        x_out = self.decode(pred)
        return x_out.detach().numpy()

    def train(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        pred = self.forward(x)
        loss = self.loss_fn(pred, x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


if __name__ == '__main__':
    model = Autoencoder(50)
    rand_vals = np.random.random((100, 50))
    for i in range(100):
        rand_int = np.random.randint(5, 49)
        rand_vals[i, rand_int:] = 0

    for _ in range(5000):
        model.train(rand_vals)
        model_out = model.test(rand_vals)
        print(np.mean(np.sum(abs(rand_vals - model.test(rand_vals)), axis=1))/50)

    # for i, val in enumerate(rand_vals):
    #     test_out = model.test(val)
    #     print(test_out)
    #     print(rand_vals[i])
    #     print(sum(abs(test_out - rand_vals[i])))

