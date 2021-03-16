import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datareaders import TensorLoaderSynthetic
from architectures.variational_autoencoder import VariationalAutoencoder
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def train_and_valid_split(x, valid_size):
    num_train = len(x)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def train_curve(original, new, method, path=None):
    plt.semilogy(original, label='VAE-loss per epoch (train)')
    plt.semilogy(new, label='VAE-loss per epoch (validation)')
    plt.legend(['VAE-loss per epoch (train)', 'VAE-loss per epoch (validation)'])
    plt.title(str(method) + ': VAE-loss per epoch on training data')
    plt.xlabel('epochs')
    plt.ylabel('VAE-loss')
    if path != None:
        plt.savefig(path)
    plt.show()

def vae_loss_fn(x, recon_x, z, mu, logvar):
    MSE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_vae(model, x, epochs, learning_rate, model_name, loss_fn=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 0.001 - 0.0001
    mse_per_epoch_train, mse_per_epoch_valid = [], []
    for epoch in range(1, epochs + 1):
        train_loss_per_epoch, valid_loss_per_epoch, valid_size = [], [], 0.3
        train_idx, valid_idx = train_and_valid_split(x, valid_size)

        # training loop
        model.train()
        for j in train_idx:
            x_i = x[j]
            optimizer.zero_grad()
            x_rec, mu, logvar = model.forward(x_i)
            z = model.reparameterize(mu, logvar)
            loss = vae_loss_fn(x_i, x_rec, z, mu, logvar)
            train_loss_per_epoch.append(loss.item())
            loss.backward()  # Backward pass
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()
        mse_per_epoch_train.append(np.array(train_loss_per_epoch).mean())

        # validation loop
        model.eval()
        for j in valid_idx:
            x_i = x[j]
            x_rec, mu, logvar = model.forward(x_i)
            z = model.reparameterize(mu, logvar)
            loss = vae_loss_fn(x_i, x_rec, z, mu, logvar)
            valid_loss_per_epoch.append(loss.item())
        mse_per_epoch_valid.append(np.array(valid_loss_per_epoch).mean())

        if epoch % 100 == 0:  # print every 50th epoch
            print('epoch : ' + str(epoch))
            print(f"loss_mean train : {np.array(train_loss_per_epoch).mean():.10f}")
            print(f"loss_mean valid : {np.array(valid_loss_per_epoch).mean():.10f}")
    torch.save(model.state_dict(), model_name)
    return mse_per_epoch_train, mse_per_epoch_valid


def get_syntheticdata_trainset():
    train_path, test_path, labels_path, error_dimensions_path = './data/Train01.txt', './data/Test01.txt', './data/Labels01.txt', './data/ErrorDimensions01.txt'
    dlhs = TensorLoaderSynthetic(train_path, test_path, labels_path, error_dimensions_path)
    X_train = []
    for i in range(len(dlhs.data_train)):
        X_train.append(dlhs.data_train[i]['time_serie'])
    return X_train

def train_variational_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # optimization

    feature_nr, hidden_size, latent_size, epochs, learning_rate = 20, 40, 12, 2000, 0.001  # define parameters for the neural network

    model = VariationalAutoencoder(feature_nr, hidden_size, latent_size).to(device)  # initialize the neural network
    X_train = get_syntheticdata_trainset() # get training dataset

    # train the model
    start = timer()
    mse_per_epoch, mse_per_epoch_valid = train_vae(model, X_train, epochs, learning_rate, './models/variational_ae_synthetic.pt')
    end = timer()
    # time and final loss
    print('AE train time (' + str(latent_size) + ' dims): ', end - start, 's')
    print('number of epochs: ', epochs)
    print('final loss: ', mse_per_epoch)

    #plot training curve
    train_curve(mse_per_epoch, mse_per_epoch_valid, 'Variational-AutoEncoder', './images/synthetic_data_variational_ae.png')


if __name__ == "__main__":
    train_variational_ae()
