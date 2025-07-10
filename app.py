import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import json

from tensorflow import keras
from keras import layers
from keras.optimizers import Adam 
from tqdm import trange
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.stats import entropy, wasserstein_distance
from json_append import append_to_json_array, clear_json_file_array

clear_json_file_array("KL_W_Info.json")
clear_json_file_array("synthetic.json")

# Metric tracking lists
kl_divergences = []
wasserstein_distances = []
epoch_list = []

df = pd.read_csv("dielectron.csv")

# Select needed features
features = ["pt1", "eta1", "phi1", "E1"]  # Make sure CSV includes energy
df = df[features].dropna()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Calculate px, py, pz
df["px"] = df["pt1"] * np.cos(df["phi1"])
df["py"] = df["pt1"] * np.sin(df["phi1"])
df["pz"] = df["pt1"] * np.sinh(df["eta1"])

# Calculate invariant mass
df["mass"] = np.sqrt(np.maximum(df["E1"]**2 - df["px"]**2 - df["py"]**2 - df["pz"]**2, 0))

# Final features for GAN training
gan_features = ["pt1", "eta1", "phi1", "mass"]
X = df[gan_features].values

# Normalize to [0, 1]
denom = X.max(axis=0) - X.min(axis=0)
denom[denom == 0] = 1
X = (X - X.min(axis=0)) / denom

# Define dimensions
data_dim = X.shape[1]
# Set parameters
latent_dim = 16
batch_size = 64
epochs = 25000

# Build Generator
gen = tf.keras.Sequential([
    layers.Dense(128, input_shape=(latent_dim,)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dense(data_dim, activation="sigmoid")
])

# Discriminator
disc = tf.keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(data_dim,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])
disc.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.00005))

# GAN (Generator + Discriminator)
disc.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = disc(gen(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

real_data = df.sample(batch_size)
noise = np.random.normal(size=(batch_size, latent_dim))
generated_data = gen.predict(noise)

#Modeling Checkpoints
best_kl = float('inf')
best_gen_weights = None

# Training loop with live evaluation plots
for epoch in trange(epochs):
    # Real samples
    idx = np.random.randint(0, X.shape[0], batch_size)
    real = X[idx]

    # Fake samples
    noise = np.random.normal(size=(batch_size, latent_dim))
    fake = gen.predict(noise, verbose=0)

    # Discriminator labels with smoothing and noise
    y_real = np.ones((batch_size, 1)) * 0.9
    y_fake = np.random.uniform(0, 0.1, (batch_size, 1))

    # Train Discriminator
    disc.trainable = True
    disc.train_on_batch(real, y_real)
    disc.train_on_batch(fake, y_fake)

    # Train Generator
    noise = np.random.normal(size=(batch_size, latent_dim))
    y_gen = np.ones((batch_size, 1))
    disc.trainable = False
    gan.train_on_batch(noise, y_gen)

    # KL/Wasserstein Block
    if epoch % 1000 == 0:
        # Sample real data
        real_data = df.sample(batch_size)

        real_pt = real_data["pt1"].values
        pt_min = df["pt1"].min()
        pt_max = df["pt1"].max()

        # Normalize real_pt to match fake_pt's scale
        real_pt = (real_pt - pt_min) / (pt_max - pt_min)
        real_pt = np.clip(real_pt, 0, 1)

        # Generate new fake data
        noise = np.random.normal(size=(batch_size, latent_dim))
        generated_data = gen.predict(noise)
        fake_pt = generated_data[:, 0]

        # Histogram comparison
        bins = np.linspace(0, 1, 50)
        real_hist, _ = np.histogram(real_pt, bins=bins, density=True)
        fake_hist, _ = np.histogram(fake_pt, bins=bins, density=True)

        # Epsilon normalize before calculating KL
        bins = np.linspace(0, 1, 50)
        real_hist, _ = np.histogram(real_pt, bins=bins, density=True)
        fake_hist, _ = np.histogram(fake_pt, bins=bins, density=True)

        # Small epsilon to avoid log(0)
        epsilon = 1e-10
        real_hist = real_hist + epsilon
        fake_hist = fake_hist + epsilon

        real_hist /= np.sum(real_hist)
        fake_hist /= np.sum(fake_hist)

        # Metrics
        kl = entropy(real_hist, fake_hist, base=2) #base-2 for bits
        w_dist = wasserstein_distance(real_pt, fake_pt)

        kl_divergences.append(kl)
        wasserstein_distances.append(w_dist)
        epoch_list.append(epoch)

        print(f"Epoch {epoch} | KL: {kl:.4f} | W: {w_dist:.4f}")
        
        # Printing KL & W data into a JSON file
        data = {
            "Epoch" : f"{epoch}",
            "KL Divergences" : f"{kl:.4f}",
            "Wsserstein Distances" : f"{w_dist:.4f}"
        }

        append_to_json_array("KL_W_Info.json", data)


    # Plot progress
    if epoch % 5000 == 0:
        synthetic = gen.predict(np.random.normal(size=(5000, latent_dim)), verbose=0)

        print("Fake pt summary:")
        print("Min:", synthetic[:, 0].min())
        print("Max:", synthetic[:, 0].max())
        print("Mean:", synthetic[:, 0].mean())
        print("Std:", synthetic[:, 0].std())

        gen_loss = gan.train_on_batch(noise, y_gen)
        print(f"Epoch {epoch}, Generator Loss: {gen_loss:.4f}")
        
        # Printing synthetic info into JSON file
        data = {
            "Epoch" : f"{epoch}",
            "Min" : f"{synthetic[:, 0].min()}",
            "Max" : f"{synthetic[:, 0].max()}",
            "Mean" : f"{synthetic[:, 0].mean()}",
            "STD" : f"{synthetic[:, 0].std()}",
            "Generator Loss" : f"{gen_loss:.4f}"
        }

        append_to_json_array("synthetic.json", data)

        plt.figure(figsize=(6, 4))
        plt.hist(X[:, 0], bins=30, alpha=0.5, label="Real (pt)", density=True)
        plt.hist(synthetic[:, 0], bins=30, alpha=0.5, label="Fake (pt)", density=True)
        plt.title(f"Feature Distribution at Epoch {epoch}")
        plt.xlabel("Feature 1 (e.g., pt)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"hist_epoch_{epoch}.png", dpi=300)
        plt.show()

    #Limit Epochs Based on Metrics
    if epoch > 20000 and kl_divergences[-1] > 1.0:
        print("KL divergence too high, stopping early.")
        break

if best_gen_weights is not None:
    gen.set_weights(best_gen_weights)
    gen.save("best_generator_model.h5")
    print("Best generator saved to best_generator_model.h5")

    disc.save("best_discriminator_model.h5")

# Plotting KL Divergence and Wasserstein Distance
plt.figure(figsize=(8, 4))
plt.plot(epoch_list, kl_divergences, label='KL Divergence')
plt.plot(epoch_list, wasserstein_distances, label='Wasserstein Distance')
plt.xlabel('Epoch')
plt.ylabel('Divergence / Distance')
plt.title('GAN Distribution Divergence Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("GAN_Distribution.png", dpi=300)
plt.show()

#Graphing Pairwise Feature Distributions
df_corr = df[gan_features].copy()
corr_matrix = df_corr.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.savefig(f"Feature Correlation Matrix.png", dpi=300)
plt.show()

sns.pairplot(df[gan_features])
plt.suptitle("Pairwise Feature Distributions", y=1.02)
plt.savefig(f"Pairwise Feature Distributions.png", dpi=300)
plt.show()

#2D Histograms (Heatmaps) of Feature Pairs
# Generate new synthetic data for comparison
noise = np.random.normal(size=(5000, latent_dim))
synthetic = gen.predict(noise)

# Denormalize synthetic data to original scale
X_min = df[gan_features].min().values
X_max = df[gan_features].max().values
synthetic = synthetic * (X_max - X_min) + X_min

# Convert real data to array for plotting
real_eval = df[gan_features].values

# Plot 2D histogram: pt vs eta
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].hist2d(real_eval[:, 0], real_eval[:, 1], bins=50, cmap='Blues')
axs[0].set_title("Real pt vs eta")
axs[0].set_xlabel("pt")
axs[0].set_ylabel("eta")

axs[1].hist2d(synthetic[:, 0], synthetic[:, 1], bins=50, cmap='Reds')
axs[1].set_title("Fake pt vs eta")
axs[1].set_xlabel("pt")
axs[1].set_ylabel("eta")

plt.tight_layout()
plt.savefig("pt_vs_eta_2D_hist.png", dpi=300)
plt.show()


#3D Scatter Plot of pt, eta, and mass
fig = plt.figure(figsize=(10, 5))

# Real
ax = fig.add_subplot(121, projection='3d')
ax.scatter(real_eval[:, 0], real_eval[:, 1], real_eval[:, 3], alpha=0.4, c='blue')
ax.set_title("Real: pt vs eta vs mass")
ax.set_xlabel("pt")
ax.set_ylabel("eta")
ax.set_zlabel("mass")

# Fake
ax = fig.add_subplot(122, projection='3d')
ax.scatter(synthetic[:, 0], synthetic[:, 1], synthetic[:, 3], alpha=0.4, c='red')
ax.set_title("Fake: pt vs eta vs mass")
ax.set_xlabel("pt")
ax.set_ylabel("eta")
ax.set_zlabel("mass")

plt.tight_layout()
plt.savefig("3D_scatter_pt_eta_mass.png", dpi=300)
plt.show()

gen.save("generator_model.h5")
disc.save("discriminator_model.h5")