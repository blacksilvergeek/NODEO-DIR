# Reproduce NODEO-DIR
[CVPR 2022] NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration

group members


# Introduction



# Method

## Reproduce Figure
### Prapare data
**Objective**: The objective of this section is to reproduce Figure X from the original paper. This figure shows...

![Figure X from the original paper](https://i.imgur.com/SiW8FrQ.png)

**Data choice**: We will use the OASIS dataset for this reproduction. Specifically, we will choose the 50th slice of subject 1 and subject 2, as shown in the following images:

![Slice 50 of subject 1,2 from the OASIS dataset](https://i.imgur.com/lRdn3tP.png)


To obtain these images, we first downloaded the OASIS dataset from the official website and then used a Python library to extract the relevant slices from the MRI scans. We selected slice 50 because it was identified as middle slice in the brain.

### Code Implementation

The original implementation at https://github.com/yifannnwu/NODEO-DIR only provides 3D methods, so we need to implement the 2D version from scratch. This involves adapting the loss and network functions to the 2D setting.



#### Adapting the Network Architecture
This is the original implementation of the network. 


```python

class BrainNet(ODEF):
    def __init__(self, img_sz, smoothing_kernel, smoothing_win, smoothing_pass, ds, bs):
        super(BrainNet, self).__init__()
        padding_mode = 'replicate'
        bias = True
        self.ds = ds
        self.bs = bs
        self.img_sz = img_sz
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_pass = smoothing_pass
        # self.enc_conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.bottleneck_sz = int(math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds)) * math.ceil(img_sz[2] / pow(2, self.ds)))
        self.lin1 = nn.Linear(864, self.bs, bias=bias)
        self.lin2 = nn.Linear(self.bs, self.bottleneck_sz * 3, bias=bias)
        self.relu = nn.ReLU()

        # Create smoothing kernels
        if self.smoothing_kernel == 'AK':
            self.sk = AveragingKernel(win=smoothing_win)
        else:
            self.sk = GaussianKernel(win=smoothing_win, nsig=0.1)

    def forward(self, x):
        imgx = self.img_sz[0]
        imgy = self.img_sz[1]
        imgz = self.img_sz[2]
        # x = self.relu(self.enc_conv1(x))
        x = F.interpolate(x, scale_factor=0.5, mode='trilinear')  # Optional to downsample the image
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.enc_conv6(x)
        x = x.view(-1)
        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        x = x.view(1, 3, int(math.ceil(imgx / pow(2, self.ds))), int(math.ceil(imgy / pow(2, self.ds))), int(math.ceil(imgz / pow(2, self.ds))))
        for _ in range(self.ds):
            x = F.upsample(x, scale_factor=2, mode='trilinear')
        # Apply Gaussian/Averaging smoothing
        for _ in range(self.smoothing_pass):
            if self.smoothing_kernel == 'AK':
                x = self.sk(x)
            else:

```

Modifying it is relatively simple, just replace the convolution with 2D, and remove the last dimension when dealing with operations involving dimensions. In addition, there is an implicit problem of reproducing the kernel here. In fact, since we don't use Gaussian, we only need to reproduce AK (Average Kernel), which is equivalent to AvgPool2d.

```python

class BrainNet_2D(ODEF):
    def __init__(self, img_sz, smoothing_kernel, smoothing_win, smoothing_pass, ds, bs):
        super(BrainNet_2D, self).__init__()
        padding_mode = 'replicate'
        bias = True
        self.ds = ds
        self.bs = bs
        self.img_sz = img_sz
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_win = smoothing_win
        self.smoothing_pass = smoothing_pass
        # self.enc_conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.bottleneck_sz = int(
            math.ceil(img_sz[0] / pow(2, self.ds)) * math.ceil(img_sz[1] / pow(2, self.ds)))

        # why 864? I think it shoudl be 32*3*3*3, so for 2d - I don't know neither lol
        self.lin1 = nn.Linear(288, self.bs, bias=bias)
        self.lin2 = nn.Linear(self.bs, self.bottleneck_sz * 2, bias=bias)
        self.relu = nn.ReLU()

        # Create smoothing kernels - to be updated for 2D 
        if self.smoothing_kernel == 'AK':
            self.sk = nn.AvgPool2d(kernel_size = smoothing_win, stride = 1)
        elif self.smoothing_kernel == 'GK':
            self.sk = GaussianKernel_2D(win=smoothing_win, nsig=0.1)

    def forward(self, x):
        smoothing_win = self.smoothing_win
        imgx = self.img_sz[0]
        imgy = self.img_sz[1]
        
        # x = self.relu(self.enc_conv1(x))
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')  # Optional to downsample the image
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.enc_conv6(x)
        x = x.view(-1) # flatten
        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        # here only re-size?
        x = x.view(1, 2, int(math.ceil(imgx / pow(2, self.ds))), int(math.ceil(imgy / pow(2, self.ds)))
                   )
        for _ in range(self.ds):
            
            x = F.upsample(x, scale_factor=2, mode='bilinear')
            
        # Apply Gaussian/Averaging smoothing
        # print(self.sk)
        for _ in range(self.smoothing_pass):
            if self.smoothing_kernel == 'AK':
                # to do: use functional.pad and then use average without padding
                x = F.pad(x, pad = [(smoothing_win)//2, (smoothing_win)//2]*2)
                x = self.sk(x)
                #print(x.shape)
            elif self.smoothing_kernel == 'GK':
                x_x = self.sk(x[:, 0, :, :].unsqueeze(1))
                x_y = self.sk(x[:, 1, :, :].unsqueeze(1))
                
                x = torch.cat([x_x, x_y], 1)

        return x

```


#### Adapting the Loss Function


The loss function contains four components, namely loss_ncc, loss_v, loss_j, and loss_df, which represent:

- loss_sim: similarity loss
- loss_v: V magnitude loss
- loss_j: negative Jacobian loss
- loss_df: phi dphi/dx loss



The implementation of loss_sim was referenced from [losses.py](https://github.com/MingR-Ma/RFR-WWANet/blob/ee89e107b2932ad1820a27d057e3b08c45fde6f9/losses.py). As for loss_J， loss_v and loss_df, the operations were similar to those performed on the network and were not complex.

```python

def JacboianDet_2D(J):
    # get type of J
    _type = type(J)
    if _type == np.ndarray:
        J = torch.from_numpy(J).float()
    if J.size(-1) != 2:
        J = J.permute(0, 2, 3, 1)
    J = J + 1
    J = J / 2.
    # print(J)

    scale_factor = torch.tensor([J.size(1), J.size(2)]).to(J).view(1, 1, 1, 2) * 1.
    J = J * scale_factor


    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]
    
    J_det = dx[..., 0] * dy[..., 1] - dy[..., 0] * dx[..., 1]
    # if _type == np.ndarray:
    #     # convert back to numpy
    #     J_det = J_det.numpy()

    return J_det

def neg_Jdet_loss_2D(J):
    Jdet = JacboianDet_2D(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet ** 2)


def neg_Jdet_loss(J):
    Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet ** 2)

def magnitude_loss_2D(all_v):
    all_v_x_2 = all_v[:, :, 0, :, :] * all_v[:, :, 0, :, :]
    all_v_y_2 = all_v[:, :, 1, :, :] * all_v[:, :, 1, :, :]
    #all_v_z_2 = all_v[:, :, 2, :, :, :] * all_v[:, :, 2, :, :, :]
    all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2)
    return all_v_magnitude

```

Here, an additional conversion to torch.tensor is written in the calculation of J_det, which is mainly for the convenience of drawing the graph later







# Result 



## Reproduce
<center>Both K and J</center>

![Both K and J](https://i.imgur.com/iLxyd9J.png)

<center>Only K</center>

![Only K](https://i.imgur.com/GgnOXeE.png)

<center>Only J</center>

![Only J](https://i.imgur.com/Ctibsz5.png)

<center>None</center>

![None](https://i.imgur.com/aemUjUf.png)


## New data

## hyperparamter check

# Conclusion





# Usage
Run Registration.py

# Label Index 
OASIS: [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]

CANDI: [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]

# Citation
@inproceedings{wu2022nodeo,
  title={Nodeo: A neural ordinary differential equation based optimization framework for deformable image registration},
  author={Wu, Yifan and Jiahao, Tom Z and Wang, Jiancong and Yushkevich, Paul A and Hsieh, M Ani and Gee, James C},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20804--20813},
  year={2022}
}
