#####################################################################
# Import libraries
#####################################################################
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import kornia  # For SSIM and Gaussian blur

#####################################################################
# Additional libraries for FID & PSNR
#####################################################################
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import peak_signal_noise_ratio

#####################################################################
# Training parameters
#####################################################################
batch_size = 256
lr = 0.0002
train_epoch = 20

optimiser = "BCEMSE"
lambda_term = 10

Test_Name = f"CELEBA_{train_epoch}Epoch_{optimiser}Optim"
Dataset = "CELEBA"
File_Name = f"{Dataset}_DCGAN"
img_size = 64

print("\n\n#####################################################################")
print(f"Script executing: {Test_Name} on {Dataset} dataset")
print(f"File Name: {File_Name} will be used to save variables")
print("#####################################################################\n")
print("1.   Libraries loaded and torch seed to be set.")

#####################################################################
# Device & Seeding
#####################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if device.type == "cpu":
    print("2.   CPU active and selected.")
else:
    print("2.   GPU active and selected.")

#####################################################################
# FID & PSNR Setup
#####################################################################
fid_metric = FrechetInceptionDistance(feature=2048).to(device)

def float_to_uint8(tensor_float: torch.Tensor, assume_neg1_to1=True) -> torch.Tensor:
    """
    Converts float images from [-1,1] (or [0,1]) to uint8 [0..255].
    If images are in [-1,1], set assume_neg1_to1=True.
    """
    if assume_neg1_to1:
        # [-1,1] -> [0,1]
        tensor_float = (tensor_float * 0.5) + 0.5
    tensor_255 = tensor_float * 255.0
    tensor_uint8 = tensor_255.clamp(0, 255).to(torch.uint8)
    return tensor_uint8

#####################################################################
# DMSE (Disparity Weighted MSE)
#####################################################################
class DMSE(nn.Module):
    def __init__(self):
        super(DMSE, self).__init__()
        
    def forward(self, fakes, reals, gamma, sigma):
        fakes = fakes.to(device)
        reals = reals.to(device)
        
        # Disparities
        x_disparity = torch.diff(reals, dim=2, append=reals[:, :, -1:, :])
        y_disparity = torch.diff(reals, dim=3, append=reals[:, :, :, -1:])
        disparities = torch.sqrt(x_disparity**2 + y_disparity**2)
        
        # Gaussian blur
        gaussian_blur = kornia.filters.GaussianBlur2d(
            (int(sigma*6+1), int(sigma*6+1)), (sigma, sigma)
        ).to(device)
        
        blurred_disparities = gaussian_blur(disparities)
        offset_disparities = blurred_disparities ** gamma
        
        # Normalize [0,1]
        epsilon = 1e-8
        min_val = offset_disparities.min()
        max_val = offset_disparities.max()
        total_disp_norm = (offset_disparities - min_val) / (max_val - min_val + epsilon)
        
        # Weight in [0.2, 1.0]
        weight = total_disp_norm * 0.8 + 0.2
        loss = torch.mean(weight * (reals - fakes) ** 2)
        return loss

print("3.   Custom loss function defined.")

#####################################################################
# Standard Losses
#####################################################################
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
SSIM_loss = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
DMSE_loss = DMSE()

print("4.   All loss functions defined and loaded.")

#####################################################################
# DCGAN Weight Initialization
#####################################################################
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

print("5.   DCGAN weight initialisations defined.")

#####################################################################
# Generator
#####################################################################
class ConditionalGenerator(nn.Module):
    def __init__(self, d=64, num_classes=2, embedding_dim=50):
        super(ConditionalGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        self.deconv1 = nn.ConvTranspose2d(100 + embedding_dim, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)  # Output channels: 3 for RGB
        
    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_emb(labels)  # [batch_size, embedding_dim]
        x = torch.cat([noise, label_embedding], dim=1)  # [batch_size, 100+embedding_dim]
        x = x.view(x.size(0), -1, 1, 1)                 # [N, (100+emb_dim),1,1]
        
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))  # -> [N,3,64,64]
        return x

print("6.   Generator structure defined.")

#####################################################################
# Discriminator
#####################################################################
class ConditionalDiscriminator(nn.Module):
    def __init__(self, d=64, num_classes=2, embedding_dim=50, img_size=64):
        super(ConditionalDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        self.conv1 = nn.Conv2d(3 + embedding_dim, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
    
    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img, labels):
        # [N,3,H,W] for img, [N] for labels
        label_embedding = self.label_emb(labels)  # [N,embedding_dim]
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)  # [N,embedding_dim,1,1]
        label_embedding = label_embedding.repeat(1,1,img.size(2),img.size(3))  # [N,embedding_dim,H,W]
        
        x = torch.cat([img, label_embedding], dim=1)  # [N, 3+embedding_dim, H, W]
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))  # [N,1,?,?]
        
        x = x.view(x.size(0), -1)  # [N,1]
        return x

print("7.   Discriminator structure defined.")

#####################################################################
# Fixed noise for image generation
#####################################################################
fixed_z_ = torch.randn((25, 100)).to(device)
print("8.   Fixed noise vector defined.")

#####################################################################
# Visualization
#####################################################################
def show_result_conditional(num_epoch, show=False, save=False, path='result.png', labels=None):
    with torch.no_grad():
        G.eval()
        if labels is None:
            labels = torch.randint(0, 2, (25,), dtype=torch.long).to(device)
        else:
            labels = labels.to(device)
        
        z_ = torch.randn((25, 100)).to(device)
        fake_images = G(z_, labels)  # [25,3,64,64]
        G.train()
    
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i,j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    
    for k in range(25):
        i = k // 5
        j = k % 5
        
        # Convert [-1,1] -> [0,1]
        img = (fake_images[k].cpu().numpy().transpose(1,2,0) + 1)/2
        img = np.clip(img,0,1)
        
        ax[i,j].imshow(img)
        ax[i,j].set_title(str(labels[k].item()), fontsize=8)
    
    label = f'Epoch {num_epoch}'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    
    if show:
        plt.show()
    else:
        plt.close()

print("9.   Image generation function defined.")

#####################################################################
# Plotting
#####################################################################
def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses_BCE']))
    y1 = hist['D_losses_BCE']
    y2 = hist['G_losses_BCE']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

print("10.  Training plot function defined.")

#####################################################################
# CelebA Dataset Setup
#####################################################################
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         std=(0.5,0.5,0.5))
])

chosen_attribute = 'Smiling'
attribute_idx = 31  # Index for 'Smiling' in the CelebA attributes

def extract_labels(targets: torch.Tensor, attribute_idx: int):
    """
    Extract a chosen attribute from CelebA's 40-dim attributes.
    'targets' shape: [batch_size, 40] with values in {-1,1}.
    Convert { -1, +1 } -> {0,1}.
    """
    labels = targets[:, attribute_idx]
    labels = (labels + 1)/2
    return labels.long().squeeze()

# Data loader (training)
train_loader = torch.utils.data.DataLoader(
    datasets.CelebA(
        './data/celeba',
        split='train',
        target_type='attr',
        download=True,
        transform=transform
    ),
    batch_size=batch_size, shuffle=True,
    num_workers=4, drop_last=True
)

# Data loader (testing)
test_loader = torch.utils.data.DataLoader(
    datasets.CelebA(
        './data/celeba',
        split='test',
        target_type='attr',
        download=True,
        transform=transform
    ),
    batch_size=batch_size, shuffle=True,
    num_workers=4, drop_last=True
)

print("11.  Dataset and Data loaders defined.")

#####################################################################
# Instantiate Networks
#####################################################################
G = ConditionalGenerator(d=64, num_classes=2, embedding_dim=50)
D = ConditionalDiscriminator(d=64, num_classes=2, embedding_dim=50, img_size=64)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

G.to(device)
D.to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

print("12.  Network and optimisers defined.")

#####################################################################
# Results folder
#####################################################################
if not os.path.isdir(f'{Test_Name}_results'):
    os.mkdir(f'{Test_Name}_results')
if not os.path.isdir(f'{Test_Name}_results/Random_results'):
    os.mkdir(f'{Test_Name}_results/Random_results')
if not os.path.isdir(f'{Test_Name}_results/Fixed_results'):
    os.mkdir(f'{Test_Name}_results/Fixed_results')

print("13.  Results folder created.")

#####################################################################
# Metric Tracking
#####################################################################
train_hist = {
    'D_losses_BCE': [],
    'G_losses_BCE': [],
    'G_losses_MSE': [],
    'G_losses_SSIM': [],
    'G_losses_DMSE': [],
    'FID': [],
    'PSNR': [],
    'per_epoch_ptimes': [],
    'total_ptime': []
}

test_hist = {
    'D_losses_BCE': [],
    'G_losses_BCE': [],
    'G_losses_MSE': [],
    'G_losses_SSIM': [],
    'G_losses_DMSE': [],
    'FID': [],
    'PSNR': [],
    'per_epoch_ptimes': [],
    'total_ptime': []
}

epoch_end_indices = []

print("14.  Training and testing metric trackers defined.")

#####################################################################
# Obtain sample images for each label if needed
#####################################################################
train_dataset = train_loader.dataset
target_images = {}

for label in range(2):
    found_sample = False
    for img, tgt_attrs in train_dataset:
        if not isinstance(tgt_attrs, torch.Tensor):
            tgt_attrs = torch.tensor(tgt_attrs, dtype=torch.float32)
        extracted_lbl = extract_labels(tgt_attrs.unsqueeze(0), attribute_idx).item()
        if extracted_lbl == label:
            target_images[label] = img.unsqueeze(0).to(device)
            found_sample = True
            break
    if not found_sample:
        print(f"Warning: Could not find sample with label={label} in the dataset.")

print("15.  Target images defined (if needed).")

print("\nTraining start!...\n")

#####################################################################
# Training
#####################################################################
start_time = time.time()

for epoch in range(train_epoch):
    epoch_start_time = time.time()
    
    # Reset FID aggregator each epoch
    fid_metric.reset()
    
    ########################################################
    # Training loop
    ########################################################
    for batch_idx, (x_, attr) in enumerate(train_loader):
        x_ = x_.to(device)   # [N,3,64,64]
        labels = extract_labels(attr, attribute_idx).to(device)  # [N], 0 or 1
        mini_batch = x_.size(0)
        
        y_real_ = torch.ones(mini_batch,1).to(device)
        y_fake_ = torch.zeros(mini_batch,1).to(device)
        
        # 1) Train Discriminator
        D_optimizer.zero_grad()
        
        D_real = D(x_, labels)
        D_real_loss = BCE_loss(D_real, y_real_)
        
        z = torch.randn(mini_batch, 100).to(device)
        fake_labels = torch.randint(0,2,(mini_batch,),dtype=torch.long).to(device)
        G_fake = G(z, fake_labels)
        
        D_fake = D(G_fake.detach(), fake_labels)
        D_fake_loss = BCE_loss(D_fake, y_fake_)
        
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        
        # 2) Train Generator
        G_optimizer.zero_grad()
        G_fake = G(z, fake_labels)
        D_fake = D(G_fake, fake_labels)
        G_BCE_loss = BCE_loss(D_fake, y_real_)
        
        # Additional losses
        # Here, we pick a "target" image based on the same fake_labels
        # (this is optional if you want reconstruction-based guidance)
        # If you want each generated sample to match a single reference image per label:
        # target = torch.cat([target_images[lbl.item()] for lbl in fake_labels], dim=0)
        # But if the dataset is large, you might skip exact per-sample matching.
        
        # For demonstration, we just do standard MSE w.r.t x_, but that might not
        # semantically match if the label is different from the original x_.
        G_MSE_loss  = MSE_loss(G_fake, x_)
        G_SSIM_loss = SSIM_loss(G_fake, x_)
        G_DMSE_loss = DMSE_loss(G_fake, x_, gamma=1.0, sigma=2.0)
        
        # PSNR
        psnr_value = peak_signal_noise_ratio(G_fake, x_, data_range=2.0)
        
        # FID requires uint8 images in [0..255]
        with torch.no_grad():
            # real
            x_uint8 = float_to_uint8(x_, assume_neg1_to1=True)
            # fake
            fake_uint8 = float_to_uint8(G_fake, assume_neg1_to1=True)
            
            fid_metric.update(x_uint8, real=True)
            fid_metric.update(fake_uint8, real=False)
        
        # Combined generator loss (simple BCE here)
        G_loss = G_BCE_loss + lambda_term * G_MSE_loss
        G_loss.backward()
        G_optimizer.step()
        
        # Store training losses
        train_hist['D_losses_BCE'].append(D_loss.item())
        train_hist['G_losses_BCE'].append(G_BCE_loss.item())
        train_hist['G_losses_MSE'].append(G_MSE_loss.item())
        train_hist['G_losses_SSIM'].append(G_SSIM_loss.item())
        train_hist['G_losses_DMSE'].append(G_DMSE_loss.item())
        train_hist['PSNR'].append(psnr_value.item())
    
    ########################################################
    # Testing loop
    ########################################################
    for batch_idx, (x_, attr) in enumerate(test_loader):
        x_ = x_.to(device)
        labels = extract_labels(attr, attribute_idx).to(device)
        mini_batch = x_.size(0)
        
        y_real_ = torch.ones(mini_batch,1).to(device)
        y_fake_ = torch.zeros(mini_batch,1).to(device)
        
        # Discriminator test
        D_real = D(x_, labels)
        D_real_loss = BCE_loss(D_real, y_real_)
        
        z = torch.randn(mini_batch, 100).to(device)
        fake_labels = torch.randint(0,2,(mini_batch,),dtype=torch.long).to(device)
        G_fake = G(z, fake_labels)
        
        D_fake = D(G_fake.detach(), fake_labels)
        D_fake_loss = BCE_loss(D_fake, y_fake_)
        
        D_test_loss = D_real_loss + D_fake_loss
        
        # Generator test
        G_test_BCE_loss = BCE_loss(D_fake, y_real_)
        
        # Additional test losses
        G_test_MSE_loss  = MSE_loss(G_fake, x_)
        G_test_SSIM_loss = SSIM_loss(G_fake, x_)
        G_test_DMSE_loss = DMSE_loss(G_fake, x_, gamma=1.0, sigma=2.0)
        
        # Optionally track PSNR or FID in test as well
        # psnr_test_value = peak_signal_noise_ratio(G_fake, x_, data_range=2.0)
        # test_hist['PSNR'].append(psnr_test_value.item())
        
        # We won't train the generator in the test loop
        G_test_loss = G_test_BCE_loss + lambda_term * G_test_MSE_loss
        
        # Store test metrics
        test_hist['D_losses_BCE'].append(D_test_loss.item())
        test_hist['G_losses_BCE'].append(G_test_BCE_loss.item())
        test_hist['G_losses_MSE'].append(G_test_MSE_loss.item())
        test_hist['G_losses_SSIM'].append(G_test_SSIM_loss.item())
        test_hist['G_losses_DMSE'].append(G_test_DMSE_loss.item())
    
    ########################################################
    # End of epoch => Compute FID
    ########################################################
    fid_value = fid_metric.compute().item()
    fid_metric.reset()
    train_hist['FID'].append(fid_value)
    
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    epoch_end_indices.append(len(train_hist['D_losses_BCE']))
    
    # Print logging
    print('[%d/%d] - ptime: %.2f, D_loss: %.4f, G_loss: %.4f, '
          'D_test: %.4f, G_test: %.4f, FID: %.4f' % (
        (epoch + 1),
        train_epoch,
        per_epoch_ptime,
        D_loss.item(),
        G_loss.item(),
        D_test_loss.item(),
        G_test_BCE_loss.item(),
        fid_value
    ))
    
    # Save sample images
    p = f'{Test_Name}_results/Random_results/{File_Name}_' + str(epoch + 1) + '.png'
    fixed_p = f'{Test_Name}_results/Fixed_results/{File_Name}_' + str(epoch + 1) + '.png'
    
    # Some fixed labels for demonstration
    fixed_labels = torch.arange(0, 2).repeat(13)[:25].to(device)
    
    show_result_conditional(epoch+1, save=True, path=fixed_p, labels=fixed_labels)
    show_result_conditional(epoch+1, save=True, path=p, labels=None)

#####################################################################
# Finish
#####################################################################
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
test_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])),
    train_epoch, total_ptime
))
print("Training finish!... save training results")

# Save model
torch.save(G.state_dict(), f"{Test_Name}_results/generator_param.pkl")
torch.save(D.state_dict(), f"{Test_Name}_results/discriminator_param.pkl")

# Save training/test history
with open(f'{Test_Name}_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
with open(f'{Test_Name}_results/test_hist.pkl', 'wb') as f:
    pickle.dump(test_hist, f)

# Plot training curves
show_train_hist(train_hist, save=True, path=f'{Test_Name}_results/{File_Name}_train_hist.png')
show_train_hist(test_hist, save=True, path=f'{Test_Name}_results/{File_Name}_test_hist.png')

# GIF generation
images = []
for e in range(train_epoch):
    img_name = f'{Test_Name}_results/Fixed_results/{File_Name}_' + str(e + 1) + '.png'
    if os.path.exists(img_name):
        images.append(imageio.imread(img_name))
imageio.mimsave(f'{Test_Name}_results/generation_animation.gif', images, fps=5)

print("\n\nTraining complete. Results saved.\n")
