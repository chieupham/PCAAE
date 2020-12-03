from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from matplotlib.widgets import Slider, Button


# ==========================================================
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open('./data/init_code_v1', 'rb') as f:
    initcode = pickle.load(f)[0,...].reshape(1,-1)
    initcode = torch.tensor(initcode).to(device)

    
class decoder(nn.Module):
    def __init__(self, ImageChannel = 512, code_size=1):
        super(decoder, self).__init__()
        self.FC1 = nn.Linear(code_size, code_size*64)
        self.FC2 = nn.Linear(code_size*64, ImageChannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        encode_x = self.FC2(self.relu(self.FC1(x)))
        return encode_x
    
class factorVAE_D(nn.Module):
    def __init__(self, ImageChannel = 512):
        self.ImageChannel = ImageChannel
        super(factorVAE_D, self).__init__()
        self.decoder_x = decoder(code_size=5)
        
    def forward(self, x):
        decode_x = self.decoder_x(x.view(-1,5))
        return decode_x

class NormalizationLayer(nn.Module):
    def __init__(self):
        super(NormalizationLayer, self).__init__()
    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())
    
model_PCAAE_D = decoder(code_size=5).to(device)
model_bTCVAE_D = decoder(code_size=5).to(device)
model_factorVAE_D = factorVAE_D().to(device)

model_PGAN = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=cuda)
generator_PGAN = model_PGAN.netG.to(device)
for param in generator_PGAN.parameters():
    param.requires_grad = False


if __name__ == "__main__":
  
    print("Testing :")
    
    checkpoint_PCAAE_D = torch.load('weights/PCAAE_PGAN',map_location=device)
    model_PCAAE_D.load_state_dict(checkpoint_PCAAE_D['PCA_D_state_dict'])
    
    checkpoint_bTCVAE_D = torch.load('weights/bTCVAE_PGAN',map_location=device)
    model_bTCVAE_D.load_state_dict(checkpoint_bTCVAE_D['BTCVAE_D_state_dict'])

    checkpoint_factorVAE_D = torch.load('weights/factorVAE_PGAN',map_location=device)
    model_factorVAE_D.load_state_dict(checkpoint_factorVAE_D['model_D6_state_dict'])
    
    # ==============================================================================
    # ==============================================================================
   
    def generate(noise,initcode=initcode,var=0.4):
        noise = var*NormalizationLayer()(noise)
        initcode = NormalizationLayer()(initcode)
        noise_temp = (noise + initcode)
        
        with torch.no_grad():
            generated_images = model_PGAN.test(noise_temp)
        grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
        return grid.permute(1, 2, 0).cpu().numpy()
           
    # ==============================================================================
    # ==============================================================================
   
    fig, ax = plt.subplots(figsize=(20,10))
    ax.axis('off')
    offset = 5
    ax1 = fig.add_subplot(1,3,3, xticklabels=[], yticklabels=[])
    ax1.set_position([0.55,0.45, 0.5, 0.5])
    imgrecon1 = plt.imshow(generate(model_PCAAE_D(torch.zeros(1,5).to(device))))
    ax1.title.set_text("Generated image of PGAN from \n the latent space of PCAAE:")

    ax2 = fig.add_subplot(1,3,2, xticklabels=[], yticklabels=[])
    ax2.set_position([0.25,0.45, 0.5, 0.5])
    imgrecon2 = plt.imshow(generate(model_factorVAE_D(torch.zeros(1,5).to(device))))
    ax2.title.set_text("Generated image of PGAN from \n the latent space of FactorVAE:")
    
    ax3 = fig.add_subplot(1,3,1, xticklabels=[], yticklabels=[])
    ax3.set_position([-0.05,0.45, 0.5, 0.5])
    imgrecon3 = plt.imshow(generate(model_bTCVAE_D(torch.zeros(1,5).to(device))))
    ax3.title.set_text("Generated image of PGAN from \n the latent space of b-TCVAE:")
    
    axcolor = 'lightgoldenrodyellow'
    axcode_list = []
    scode_list = []     
    h_size = 0.04
    h_size_temp = h_size
    for idx in range(5):  
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
            reconst_PCAAE_D = model_PCAAE_D(fcode)
            reconst_bTCVAE_D = model_bTCVAE_D(fcode)
            reconst_factorVAE_D = model_factorVAE_D(fcode)

        imgrecon1.set_data(generate(reconst_PCAAE_D))
        imgrecon2.set_data(generate(reconst_factorVAE_D))
        imgrecon3.set_data(generate(reconst_bTCVAE_D))
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