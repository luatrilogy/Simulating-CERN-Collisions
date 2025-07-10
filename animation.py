import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model

# Load your trained generator model
gen = load_model('generator_model.h5')  # replace with actual path
latent_dim = 16

# Generate synthetic particles from the GAN
num_particles = 100
noise = np.random.normal(size=(num_particles, latent_dim))
synthetic = gen.predict(noise)

# Extract physical features
pt = synthetic[:, 0]
eta = synthetic[:, 1]
phi = synthetic[:, 2] * 2 * np.pi  # denormalize if needed

# Convert (pt, eta, phi) → (px, py, pz)
px = pt * np.cos(phi)
py = pt * np.sin(phi)
pz = pt * np.sinh(eta)

# Normalize vectors for display
norm = np.sqrt(px**2 + py**2 + pz**2)
px /= norm
py /= norm
pz /= norm

# Set up 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()

    # Expanding detector rings
    for i in range(3):
        radius = (frame * 0.02) % 1.0 + i * 0.15
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)
        ax.plot(x, y, z, color='gray', alpha=0.3)

    # Red momentum vectors
    ax.quiver(np.zeros(num_particles), np.zeros(num_particles), np.zeros(num_particles),
              px, py, pz, length=0.6, normalize=True, color='red', alpha=0.8)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(f"GAN Particle Collision – Frame {frame}")
    ax.set_xlabel('px')
    ax.set_ylabel('py')
    ax.set_zlabel('pz')

ani = FuncAnimation(fig, update, frames=60, interval=100, blit=False)
plt.show()