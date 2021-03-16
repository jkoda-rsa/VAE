import matplotlib.pyplot as plt
import pandas as pd

from datareaders import SyntheticReader
from timeit import default_timer as timer
import torch
import numpy as np
from architectures.variational_autoencoder import VariationalAutoencoder

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def measure_baseline_time(model, model_path):
    print('time baseline measurement!')
    train_path, test_path, labels_path, error_dimensions_path = './data/Train01.txt', './data/Test01.txt', './data/Labels01.txt', './data/ErrorDimensions01.txt'
    sr = SyntheticReader(train_path, test_path, labels_path, error_dimensions_path)
    X_train = torch.tensor(sr.X_train.astype(np.float32).values)

    print("train size (original): ", sr.X_train.shape)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    start = timer()
    mu, logvar = model.encode(X_train)
    z = model.reparameterize(mu, logvar)
    X_train_latent = z.tolist()
    end = timer()
    print('number of latent dimensions:', LATENT_DIMENSIONS)
    print('VAE transform time (1000 samples): ', (end - start), 's')
    print('VAE transform time per sample: ', (end - start) / len(sr.X_train), 's')


def compare_reconstruction(model):
    print('reconstruction comparison!')
    train_path, test_path, labels_path, error_dimensions_path = './data/Train01.txt', './data/Test01.txt', './data/Labels01.txt', './data/ErrorDimensions01.txt'
    sr = SyntheticReader(train_path, test_path, labels_path, error_dimensions_path)
    X_train = torch.tensor(sr.X_train.astype(np.float32).values)

    X_reconst, mu, logvar = model.forward(X_train)
    X_reconst = np.array(X_reconst.tolist())
    df_x_rec = pd.DataFrame(X_reconst)
    df_x = pd.DataFrame(X_train)
    plt.plot(df_x[0][0:100])
    plt.plot(df_x_rec[0][0:100])
    plt.plot(df_x[1][0:100])
    plt.plot(df_x_rec[1][0:100])
    plt.plot(df_x[2][0:100])
    plt.plot(df_x_rec[2][0:100])
    plt.xlabel('timestep t')
    plt.ylabel('measurement')
    plt.title('training data: reconstruction comparison')
    plt.legend(
        ['Feature 0 (original)', 'Feature 0 (reconstructed)', 'Feature 1 (original)', 'Feature 1 (reconstructed)',
         'Feature 2 (original)', 'Feature 2 (reconstructed)'])
    plt.savefig('./images/vae_train_feature_comparison.png')
    plt.show()


def compare_sampling(model):
    print('sampling comparison!')
    train_path, test_path, labels_path, error_dimensions_path = './data/Train01.txt', './data/Test01.txt', './data/Labels01.txt', './data/ErrorDimensions01.txt'
    sr = SyntheticReader(train_path, test_path, labels_path, error_dimensions_path)
    X_train = torch.tensor(sr.X_train.astype(np.float32).values)

    X_reconst = model.sample(1000, X_train[:][999:1000])  # take last points for sampling
    df_x_rec = pd.DataFrame(np.array(X_reconst.tolist()))
    df_x = pd.DataFrame(X_train)
    plt.plot(df_x[0][0:100])
    plt.plot(df_x_rec[0][0:100])
    plt.plot(df_x[1][0:100])
    plt.plot(df_x_rec[1][0:100])
    plt.plot(df_x[2][0:100])
    plt.plot(df_x_rec[2][0:100])
    plt.xlabel('timestep t')
    plt.ylabel('measurement')
    plt.title('training data: sampling')
    plt.legend(['Feature 0 (original)', 'Feature 0 (sampled)', 'Feature 1 (original)', 'Feature 1 (sampled)',
                'Feature 2 (original)', 'Feature 2 (sampled)'])
    plt.savefig('./images/vae_sample_feature_comparison.png')
    plt.show()


if __name__ == "__main__":
    INPUT_SIZE = 20
    HIDDEN_SIZE = 40
    LATENT_DIMENSIONS = 12
    model_path = './models/variational_ae_synthetic.pt'
    model = VariationalAutoencoder(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIMENSIONS)
    measure_baseline_time(model, model_path)
    compare_reconstruction(model)
    compare_sampling(model)
