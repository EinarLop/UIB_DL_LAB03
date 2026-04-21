"""Helper functions for LAB03 — Generative Adversarial Networks.

Provides dataset utilities, loss functions, training infrastructure,
visualization, and checkpointing for GAN experiments.
"""

import os
import zipfile
import PIL
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level BCE criterion (avoid re-instantiation on every call)
_bce_criterion = nn.BCEWithLogitsLoss()


# =============================================================================
# Dataset Utilities
# =============================================================================

def extract_dataset(zip_file_path, remove_zip=True):
    """Extract a zip file containing a dataset.

    Args:
        zip_file_path (str): Path to the zip file.
        remove_zip (bool): Whether to remove the zip after extraction.

    Returns:
        str: Path to the extracted dataset folder.
    """
    dataset_folder = os.path.splitext(zip_file_path)[0]
    if os.path.exists(dataset_folder):
        print(f"Dataset folder already exists: {dataset_folder}")
        return dataset_folder

    if not os.path.exists(zip_file_path):
        raise FileNotFoundError("Zip file not found.")

    os.makedirs(dataset_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print('Extracting files...')
        zip_ref.extractall(os.path.dirname(zip_file_path))
        print('Extraction finished.')

    if not os.listdir(dataset_folder):
        raise RuntimeError(
            f"Extraction completed but {dataset_folder} is empty. "
            "The zip file's internal structure may not match expectations.")

    if remove_zip:
        os.remove(zip_file_path)
        print('Zip file removed.')

    return dataset_folder


class CustomDataset(Dataset):
    """Custom dataset for loading images from a folder.

    Args:
        img_folder (str): Path to the image folder.
        lim (int): Maximum number of images to load (-1 for all).
        transforms: Torchvision transforms to apply.
    """
    def __init__(self, img_folder, lim=-1, transforms=None):
        self.img_folder = img_folder
        self.lim = lim

        self.items = []
        self.labels = []

        for root, dirs, files in os.walk(img_folder):
            dirs.sort()
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    self.items.append(full_path)
                    self.labels.append(file)

                    if lim > 0 and len(self.items) >= lim:
                        break

            if lim > 0 and len(self.items) >= lim:
                break

        self.transforms = transforms

    def __getitem__(self, idx):
        data = PIL.Image.open(self.items[idx]).convert('RGB')
        if self.transforms:
            data = self.transforms(data)
        return data, self.labels[idx]

    def __len__(self):
        return len(self.items)


class ToScaledTensor(transforms.ToTensor):
    """Transform that converts an image to a tensor scaled to [low, high].

    By default scales to [-1, 1], which matches the tanh output range
    used in GAN generators.

    Args:
        low (float): Minimum value of the output range.
        high (float): Maximum value of the output range.
    """
    def __init__(self, low=-1, high=1):
        super().__init__()
        self.low = low
        self.high = high

    def __call__(self, img):
        tensor = super().__call__(img)
        tensor = tensor * (self.high - self.low) + self.low
        return tensor


# =============================================================================
# Network Architectures
# =============================================================================

class Generator(nn.Module):
    """DCGAN-style generator with 4 transposed convolution blocks.

    Maps a latent vector z to a 3-channel image of size img_size x img_size.

    Args:
        z_dim (int): Dimensionality of the latent space.
        d_dim (int): Base number of filters.
        img_size (int): Output image size (default 64).
    """
    def __init__(self, z_dim=100, d_dim=64, img_size=64):
        super().__init__()

        kernel_size = 4
        self.d_dim = d_dim
        n = 4  # Number of transposed conv layers

        fc_out = int(d_dim * 2**(n-1) * (img_size / (2**n))**2)
        self.i_s_fco = int(img_size / 2**n)

        pad = 1
        stride = 2
        bias = False

        def tconv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=pad,
                                      bias=bias)

        self.fc = nn.Linear(z_dim, fc_out)

        self.tconv1 = tconv(d_dim * 8, d_dim * 4)
        self.tconv2 = tconv(d_dim * 4, d_dim * 2)
        self.tconv3 = tconv(d_dim * 2, d_dim)
        self.tconv4 = tconv(d_dim, 3)

        self.bnorm1 = nn.BatchNorm2d(d_dim * 4)
        self.bnorm2 = nn.BatchNorm2d(d_dim * 2)
        self.bnorm3 = nn.BatchNorm2d(d_dim)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        b = x.size(0)
        out = self.fc(x)
        out = out.contiguous().view(b, -1, self.i_s_fco, self.i_s_fco)

        out = self.relu(self.bnorm1(self.tconv1(out)))
        out = self.relu(self.bnorm2(self.tconv2(out)))
        out = self.relu(self.bnorm3(self.tconv3(out)))
        out = self.tanh(self.tconv4(out))

        return out


# =============================================================================
# Loss Functions
# =============================================================================

def gan_loss_fcn(discr_output, **kwargs):
    """Binary cross-entropy loss for classical GAN training.

    Args:
        discr_output: Tensor of discriminator logits.
        **kwargs:
            label_type (str): "real" or "fake". Defaults to "real".
            smooth (bool): Apply label smoothing on real labels.
                Defaults to False.

    Returns:
        Tensor: The computed loss.
    """
    batch_size = discr_output.size(0)

    label_type = kwargs.get("label_type", "real").lower()
    smooth = kwargs.get("smooth", False)

    if label_type not in ("real", "fake"):
        raise ValueError(
            f"label_type must be 'real' or 'fake', got '{label_type}'")

    if label_type == "real":
        labels = torch.ones(batch_size) * (0.9 if smooth else 1.0)
    else:
        labels = torch.zeros(batch_size)

    labels = labels.to(DEVICE)

    loss = _bce_criterion(discr_output.squeeze(), labels)

    return loss


def Wasserstein_loss_fcn(input_tensor, **kwargs):
    """Wasserstein loss for WGAN training.

    The critic maximizes E[C(real)] - E[C(fake)], so:
    - For real samples: loss = -mean(scores)
    - For fake samples: loss = mean(scores)

    Args:
        input_tensor: Tensor of critic scores.
        **kwargs:
            label_type (str): "real" or "fake". Defaults to "real".

    Returns:
        Tensor: The computed loss.
    """
    label_type = kwargs.get("label_type", "real").lower()

    if label_type not in ("real", "fake"):
        raise ValueError(
            f"label_type must be 'real' or 'fake', got '{label_type}'")

    sign = -1 if label_type == 'real' else 1

    return input_tensor.mean() * sign


def penalty_fcn(real_img, fake_img, critic, gamma=10):
    """Gradient penalty for WGAN-GP.

    Enforces the 1-Lipschitz constraint by penalizing gradients that deviate
    from unit norm on interpolated samples between real and fake images.

    Args:
        real_img: Batch of real images.
        fake_img: Batch of fake images (detached).
        critic: The critic network.
        gamma (float): Penalty coefficient. Defaults to 10.

    Returns:
        Tensor: The gradient penalty term (gamma * penalty).
    """
    alpha = torch.rand(real_img.shape[0], 1, 1, 1, device=DEVICE)

    mix_images = torch.lerp(fake_img, real_img, alpha)
    mix_images.requires_grad_(True)

    mix_pred = critic(mix_images)

    gradients = torch.autograd.grad(
        inputs=mix_images,
        outputs=mix_pred,
        grad_outputs=torch.ones_like(mix_pred),
        retain_graph=True,
        create_graph=True,
        only_inputs=True)[0]

    gradients = gradients.view(len(gradients), -1)

    gradient_norm = torch.linalg.vector_norm(gradients, ord=2, dim=1)

    gp = ((gradient_norm - 1) ** 2).mean()

    return gp * gamma


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(name, epoch, config, path):
    """Save a training checkpoint (generator and discriminator/critic).

    Args:
        name (str): Checkpoint name (used in filename).
        epoch (int): Current epoch number.
        config (dict): Training configuration with 'generator', 'g_optimizer',
            'discriminator', 'd_optimizer' keys.
        path (str): Directory to save the checkpoint file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    torch.save({'epoch': epoch,
                'timestamp': timestamp,
                'gener_model_state_dict': config['generator'].state_dict(),
                'g_optimizer_state_dict': config['g_optimizer'].state_dict(),
                'discr_model_state_dict': config['discriminator'].state_dict(),
                'd_optimizer_state_dict': config['d_optimizer'].state_dict()},
               os.path.join(path, f"GAN-{name}.pkl"))


def load_checkpoint(name, config, path):
    """Load a training checkpoint.

    Args:
        name (str): Checkpoint name (used in filename).
        config (dict): Training configuration dict to update with loaded states.
        path (str): Directory containing the checkpoint file.

    Returns:
        dict or None: Dictionary with loaded components, or None if not found.
    """
    file_path = os.path.join(path, f"GAN-{name}.pkl")

    if not os.path.isfile(file_path):
        print(f"Checkpoint file {file_path} not found.")
        return None

    checkpoint = torch.load(file_path, weights_only=False)

    config['generator'].load_state_dict(checkpoint['gener_model_state_dict'])
    config['g_optimizer'].load_state_dict(checkpoint['g_optimizer_state_dict'])
    config['discriminator'].load_state_dict(checkpoint['discr_model_state_dict'])
    config['d_optimizer'].load_state_dict(checkpoint['d_optimizer_state_dict'])

    config['epoch'] = checkpoint['epoch']

    print(f"Checkpoint was saved on: "
          f"{checkpoint.get('timestamp', 'Timestamp not found')}")

    return {
        'epoch': checkpoint['epoch'],
        'generator': config['generator'],
        'g_optimizer': config['g_optimizer'],
        'discriminator': config['discriminator'],
        'd_optimizer': config['d_optimizer']}


def checkpointer(epoch, epoch_gener_loss, best_gen_loss, config,
                 save_step, starting_from=20):
    """Conditionally save a checkpoint based on loss improvement or interval.

    Checkpoints are saved if:
    - The generator loss improved (after starting_from epochs), or
    - The current epoch is a multiple of save_step.

    The checkpoint name can include a prefix via config['checkpoint_prefix']
    to avoid filename collisions between different training runs.

    Args:
        epoch (int): Current training epoch.
        epoch_gener_loss (float): Generator loss for the current epoch.
        best_gen_loss (float): Best generator loss observed so far.
        config (dict): Training configuration dict.
        save_step (int): Save interval (epochs).
        starting_from (int): Epoch after which to start saving on improvement.

    Returns:
        float: Updated best_gen_loss.
    """
    save_path = config.get('project_dir', '.')
    prefix = config.get('checkpoint_prefix', '')
    if prefix:
        prefix = f"{prefix}_"

    if epoch_gener_loss < best_gen_loss and epoch > starting_from:
        best_gen_loss = epoch_gener_loss
        print(f"New best generator loss {best_gen_loss} at epoch"
              f" {epoch}. Saving checkpoint.")
        save_checkpoint(f"{prefix}best_{epoch}", epoch, config, save_path)

    elif epoch % save_step == 0:
        print(f"Epoch {epoch}. "
              "Not best losses, but saving checkpoint anyway.")
        save_checkpoint(f"{prefix}epoch_{epoch}", epoch, config, save_path)

    return best_gen_loss


# =============================================================================
# Visualization
# =============================================================================

def show(tensor, num=25):
    """Display a 5x5 grid of 25 images from a batch tensor.

    Always displays exactly 25 images in a 5x5 grid. The ``num`` parameter
    controls which 25 images to show (the last 25 up to index ``num``).

    Args:
        tensor: Batch of image tensors (expected range [-1, 1]).
            Must contain at least 25 images.
        num (int): Upper index for image selection. Defaults to 25.
    """
    if len(tensor) < 25:
        raise ValueError(
            f"show() requires at least 25 images, got {len(tensor)}")

    num = max(25, min(num, len(tensor)))

    data = (tensor.detach().cpu() + 1) * 0.5

    grid = make_grid(data[num-25:num], nrow=5).permute(1, 2, 0)

    plt.figure(figsize=(7, 7))
    plt.imshow(grid.clip(0, 1))
    plt.show()


def visual_epoch(fake_imgs, real_imgs, gener_losses_epoch_list,
                 discr_losses_epoch_list):
    """Display generated/real images and loss curves at epoch end.

    Args:
        fake_imgs: Batch of generated images.
        real_imgs: Batch of real images.
        gener_losses_epoch_list: List of generator losses per epoch.
        discr_losses_epoch_list: List of discriminator/critic losses per epoch.
    """
    plt.close('all')
    if len(fake_imgs) >= 25:
        show(fake_imgs)
    else:
        print(f"  (Skipping image grid: batch has {len(fake_imgs)} images, need 25)")
    if len(real_imgs) >= 25:
        show(real_imgs)
    plt.figure(figsize=(10, 5))
    plt.plot(gener_losses_epoch_list, label="Generator Loss")
    plt.plot(discr_losses_epoch_list, label="D/C Loss")
    plt.legend()
    plt.show()


def diversity_score(images, n_samples=25):
    """Compute mean pairwise L2 distance among generated images.

    Higher values indicate more diverse outputs; low values suggest
    mode collapse.

    Args:
        images: Batch of image tensors (B, C, H, W).
        n_samples (int): Number of images to sample for comparison.

    Returns:
        float: Mean pairwise L2 distance.
    """
    imgs = images.detach().cpu()
    n = min(n_samples, len(imgs))

    if n < 2:
        raise ValueError(
            f"diversity_score requires at least 2 images, got {n}")

    imgs = imgs[:n].view(n, -1)  # Flatten to (n, C*H*W)

    # Compute pairwise distances using broadcasting
    diffs = imgs.unsqueeze(0) - imgs.unsqueeze(1)  # (n, n, D)
    dists = torch.linalg.vector_norm(diffs, ord=2, dim=2)  # (n, n)

    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise_dists = dists[mask]

    return pairwise_dists.mean().item()


def visual_comparison(imgs_a, imgs_b, title_a="A", title_b="B", num=25):
    """Display two sets of images side by side for comparison.

    Args:
        imgs_a: First batch of image tensors (range [-1, 1]).
        imgs_b: Second batch of image tensors (range [-1, 1]).
        title_a (str): Title for the first set.
        title_b (str): Title for the second set.
        num (int): Number of images to show from each set.
    """
    if len(imgs_a) == 0 or len(imgs_b) == 0:
        raise ValueError("visual_comparison requires non-empty image batches")

    num = min(num, len(imgs_a), len(imgs_b))
    num = max(num, 1)

    data_a = (imgs_a.detach().cpu()[:num] + 1) * 0.5
    data_b = (imgs_b.detach().cpu()[:num] + 1) * 0.5

    nrow = int(num ** 0.5)
    if nrow * nrow < num:
        nrow += 1

    grid_a = make_grid(data_a, nrow=nrow).permute(1, 2, 0)
    grid_b = make_grid(data_b, nrow=nrow).permute(1, 2, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(grid_a.clip(0, 1))
    ax1.set_title(title_a)
    ax1.axis('off')
    ax2.imshow(grid_b.clip(0, 1))
    ax2.set_title(title_b)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Training
# =============================================================================

def train(config, verbose=True):
    """Train a GAN (classical or WGAN-GP) using the provided configuration.

    The config dict must contain:
        - 'dataloader': DataLoader for training data
        - 'generator': Generator network
        - 'discriminator': Discriminator/Critic network
        - 'g_optimizer': Generator optimizer
        - 'd_optimizer': Discriminator/Critic optimizer
        - 'loss_fcn': Loss function (gan_loss_fcn or Wasserstein_loss_fcn)

    Optional config keys:
        - 'n_epochs' (int): Number of epochs. Default: 100.
        - 'z_dim' (int): Latent space dimension. Default: 100.
        - 'crit_cycles' (int): Critic training cycles per generator step.
            Default: 1.
        - 'show_step' (int): Visualization interval. Default: 25.
        - 'save_step' (int): Checkpoint interval. Default: 5.
        - 'save_starting' (int): Epoch to start saving on improvement.
            Default: 20.
        - 'penalty_fcn': Gradient penalty function. Default: no penalty.
        - 'project_dir' (str): Directory for checkpoints. Default: '.'.
        - 'epoch' (int): Last completed epoch (for resuming). Default: 0.

    Args:
        config (dict): Training configuration.
        verbose (bool): Whether to print epoch logs.

    Returns:
        tuple: (generator, discriminator, [gen_losses, disc_losses])
    """
    best_gen_loss = float('inf')

    gener_losses_epoch_list = []
    discr_losses_epoch_list = []

    n_epochs = config.get('n_epochs', 100)
    crit_cycles = config.get('crit_cycles', 1)
    z_dim = config.get('z_dim', 100)
    show_step = config.get('show_step', 25)
    save_step = config.get('save_step', 5)
    last_epoch = config.get('epoch', 0)
    save_starting = config.get('save_starting', 20)
    gp_fcn = config.get('penalty_fcn', lambda *x: 0)

    dataloader = config['dataloader']
    gener = config['generator'].to(DEVICE)
    discr = config['discriminator'].to(DEVICE)
    gener_opt = config['g_optimizer']
    discr_opt = config['d_optimizer']
    loss_fcn = config['loss_fcn']

    for epoch in range(last_epoch + 1, n_epochs + last_epoch + 1):
        epoch_gener_loss = 0.0
        epoch_discr_loss = 0.0

        num_batches = 0

        for real_imgs, _ in tqdm(dataloader):
            num_batches += 1

            current_bs = len(real_imgs)

            real_imgs = real_imgs.to(DEVICE)

            # Train discriminator/critic
            discr_loss_for_cycles = 0
            for _ in range(crit_cycles):
                discr_opt.zero_grad()

                noise = torch.randn(current_bs, z_dim, device=DEVICE)

                fake_imgs = gener(noise)

                discr_fake_pred = discr(fake_imgs.detach())
                discr_fake_loss = loss_fcn(discr_fake_pred, label_type='fake')

                discr_real_pred = discr(real_imgs)
                discr_real_loss = loss_fcn(discr_real_pred, label_type='real')

                penalty = gp_fcn(real_imgs, fake_imgs.detach(), discr)

                discr_loss = discr_fake_loss + discr_real_loss + penalty

                discr_loss_for_cycles += discr_loss.item() / crit_cycles

                discr_loss.backward()
                discr_opt.step()

            epoch_discr_loss += discr_loss_for_cycles

            # Train generator
            gener_opt.zero_grad()

            noise = torch.randn(current_bs, z_dim, device=DEVICE)
            fake_imgs = gener(noise)
            discr_fake_pred = discr(fake_imgs)

            gener_loss = loss_fcn(discr_fake_pred, label_type='real')
            epoch_gener_loss += gener_loss.item()

            gener_loss.backward()
            gener_opt.step()

        epoch_gener_loss /= num_batches
        epoch_discr_loss /= num_batches
        gener_losses_epoch_list.append(epoch_gener_loss)
        discr_losses_epoch_list.append(epoch_discr_loss)

        if verbose:
            print({'Epoch': epoch,
                   'D/C loss': epoch_discr_loss,
                   'Gen loss': epoch_gener_loss})

        # Checkpoint (capture updated best_gen_loss)
        best_gen_loss = checkpointer(
            epoch=epoch,
            epoch_gener_loss=epoch_gener_loss,
            best_gen_loss=best_gen_loss,
            config=config,
            save_step=save_step,
            starting_from=save_starting)

        if epoch % show_step == 0:
            visual_epoch(fake_imgs, real_imgs,
                         gener_losses_epoch_list,
                         discr_losses_epoch_list)

    return gener, discr, [gener_losses_epoch_list, discr_losses_epoch_list]
