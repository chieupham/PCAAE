from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
 
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
image_size = 64

# =========================================================
    
class decoder(nn.Module):
    def __init__(self, code_size=1):
        super(decoder, self).__init__()

        # Layer parameters
        kernel_size = 4
        n_chan = 1

        # Shape required to start transpose convs
        self.reshape = (code_size, 1, 1)
        
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.convT6 = nn.ConvTranspose2d(code_size, int(code_size*2), kernel_size, **cnn_kwargs)
        self.convT5 = nn.ConvTranspose2d(int(code_size*2), int(code_size*4), kernel_size, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(int(code_size*4), int(code_size*8), kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(int(code_size*8), int(code_size*16), kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(int(code_size*16), int(code_size*32), kernel_size, **cnn_kwargs)
        self.convT1 = nn.ConvTranspose2d(int(code_size*32), n_chan, kernel_size, **cnn_kwargs)
        self.leaky = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        batch_size = z.size(0)
        x = z.view(batch_size, *self.reshape)
        
        # Convolutional layers with ReLu activations
        x = self.leaky(self.convT6(x))
        x = self.leaky(self.convT5(x))
        x = self.leaky(self.convT4(x))
        x = self.leaky(self.convT3(x))
        x = self.leaky(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT1(x))

        return x

class PCAAE_D(nn.Module):
    def __init__(self, ImageChannel=1):
        super(PCAAE_D, self).__init__()
        self.decoder_x = decoder(code_size=3)
        self.ImageChannel = ImageChannel
        
    def forward(self, x):
        decode_x = self.decoder_x(x.view(-1,3))
        return decode_x.view(-1, self.ImageChannel, image_size, image_size)
    
class VAE_D(nn.Module):
    def __init__(self, ImageChannel=1):
        super(VAE_D, self).__init__()
        self.decoder_x = decoder(code_size=3)
        self.ImageChannel = ImageChannel
        
    def forward(self, x):
        decode_x = self.decoder_x(x.view(-1,3))
        return decode_x.view(-1, self.ImageChannel, image_size, image_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

class bVAE_D(nn.Module):
    def __init__(self, ImageChannel=1):
        super(bVAE_D, self).__init__()
        self.decoder_x = decoder(code_size=3)
        self.ImageChannel = ImageChannel
        
    def forward(self, x):
        decode_x = self.decoder_x(x.view(-1,3))
        return decode_x.view(-1, self.ImageChannel, image_size, image_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
model_VAE_D = VAE_D().to(device)
model_bVAE_D = bVAE_D().to(device)
model_PCAAE_D = PCAAE_D().to(device)

if __name__ == "__main__":
    print("Testing :")
    checkpoint_PCAAE = torch.load('weights/PCAAE_ellipses',map_location=device)
    model_PCAAE_D.load_state_dict(checkpoint_PCAAE['model_D3_state_dict'])       
        
    checkpoint_VAE = torch.load('weights/VAE_ellipses',map_location=device)
    model_VAE_D.load_state_dict(checkpoint_VAE['model_D3_state_dict'])
        
    checkpoint_bVAE = torch.load('weights/bVAE_ellipses',map_location=device)
    model_bVAE_D.load_state_dict(checkpoint_bVAE['model_D3_state_dict'])
    
    with torch.no_grad():   
        z = torch.zeros((1,3)).to(device)
        recon_PCAAE = model_PCAAE_D(z)
        recon_VAE = model_VAE_D(z)
        recon_bVAE = model_bVAE_D(z)
        
    # ==============================================================================
    # ==============================================================================
   
    fig, ax = plt.subplots(figsize=(20,10))
    ax.axis('off')
    offset = 1.5
    ax3 = fig.add_subplot(1,3,3, xticklabels=[], yticklabels=[])
    ax3.set_position([0.55,0.45, 0.5, 0.5])
    imgrecon3 = plt.imshow(np.squeeze(recon_PCAAE.cpu().numpy()),cmap='gray')
    ax3.title.set_text("Interpolation from the latent space of PCAAE:")
    
    ax2 = fig.add_subplot(1,3,2, xticklabels=[], yticklabels=[])
    ax2.set_position([0.25,0.45, 0.5, 0.5])
    imgrecon2 = plt.imshow(np.squeeze(recon_bVAE.cpu().numpy()),cmap='gray')
    ax2.title.set_text("Interpolation from the latent space of beta-VAE: ")

    ax1 = fig.add_subplot(1,3,1, xticklabels=[], yticklabels=[])
    ax1.set_position([-0.05,0.45, 0.5, 0.5])
    imgrecon1 = plt.imshow(np.squeeze(recon_VAE.cpu().numpy()),cmap='gray')
    ax1.title.set_text("Interpolation from the latent space of VAE: ")
    
    axcolor = 'lightgoldenrodyellow'
    axcode_list = []
    scode_list = []            
    h_size = 0.04
    h_size_temp = h_size
    for idx in range(3):  
            axcode = plt.axes([0.2, h_size*h_size_temp + h_size*idx, 0.6, h_size], facecolor=axcolor)
            axcode_list.append(axcode)
            scode_list.append(Slider(axcode, 'Component '+str(idx+1)+' :', 
                             -offset, 
                             +offset, 
                             valinit=0,
                             valstep=0.001))
    
    def update_code(val):
        fcode_list = []
        for idx in range(len(scode_list)):
            fcode_list.append(np.asarray([scode_list[idx].val])[np.newaxis,:])
    
        fcode = np.array(np.concatenate(fcode_list, axis=-1),dtype='float32')
        fcode = torch.tensor(fcode).to(device)
        with torch.no_grad():
            recon_PCAAE = model_PCAAE_D.eval()(fcode)
            recon_bVAE = model_bVAE_D.eval()(fcode)
            recon_VAE = model_VAE_D.eval()(fcode)
    
        imgrecon3.set_data(np.squeeze(recon_PCAAE[0].cpu().numpy()))
        imgrecon2.set_data(np.squeeze(recon_bVAE[0].cpu().numpy()))
        imgrecon1.set_data(np.squeeze(recon_VAE[0].cpu().numpy()))
        fig.canvas.draw_idle()
    
    for idx in range(len(scode_list)):
        scode_list[idx].on_changed(update_code)
    
    resetax = plt.axes([0.9, 0.1, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset(event):
        for idx in range(len(scode_list)):
            scode_list[idx].reset()
    button.on_clicked(reset)
    
    plt.show()