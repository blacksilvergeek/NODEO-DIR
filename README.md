# Reproduceability Project for NODEO-DIR
[CVPR 2022] NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration

Group members:

- **Yongsheng Han**
  - Contact: Y.Han-19@student.tudelft.nl
  - Student ID: 5763371
  - Main Content/Task: Reproduce results

- **Nazariy Lonyuk**
  - Contact: n.b.lonyuk@student.tudelft.nl
  - Student ID: 4596811
  - Main Content/Task: Apply model to new data and hyperparamter check

# How to Run
Run Registration_iter.py to reproduce data for table.

In main.ipynb, you can find the reproduce of figure.

# Overview

Hereby an overview of who wrote which parts:


|    Part      |   Writer          | 
| :-----------------: | :-----------------: | 
| Introduction   | Nazariy Lonyuk   | 
| Reproduction  | Yongsheng Han   | 
| New Data  | Nazariy Lonyuk   | 
| Hyperparameter Analysis  | Nazariy Lonyuk   | 
| Conclusion  | Nazariy Lonyuk   |


# Introduction

In this blog we will discuss NODEO, a framework optimized for Deformable Image Registration. Based on new techniques Wu et al created code that outperforms various benchmarks. Though the results have been shown in the paper written by Wu et al, we will reproduce the results and show how we did it based on their code repository on github. In addition to that we will apply the code on a new dataset and perform an ANOVA analysis on several hyperparameters.


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

## Reproduce table
Just just the original code it can work. All we need to do is make it possible to loop for different dataset.

Here we use ID 1 as fixed dataset, ID from 2 to 49 as moving dataset. While for the paper, they set five images with IDs 1, 10, 20, 30, 40 as the atlases, and the remaining images with IDs < 50, as moving images. Here, we change it to reduced calculation time.








# Result 



## Reproduce figure
<center>Both K and J</center>

![Both K and J](https://i.imgur.com/iLxyd9J.png)

<center>Only K</center>

![Only K](https://i.imgur.com/GgnOXeE.png)

<center>Only J</center>

![Only J](https://i.imgur.com/Ctibsz5.png)

<center>None</center>

![None](https://i.imgur.com/aemUjUf.png)

## Reproduce table
That's our result in average. The detailed data can be found in results_lambda_2.csv and results_lambda_2.5.csv

| lambda | average dice | standard derivation | $\mathcal{D}_\psi(\mathrm{x}) \leq 0\left(r^{\mathcal{D}}\right) )$ | $\mathcal{D}_\psi(\mathrm{x}) \leq 0\left(s^{\mathcal{D}}\right)$ |
|--------|--------------|---------------------|----------|----------|
| 2.5    | 0.862201     | 0.055043            | 0.0202% | 43.7181  |
| 2      | 0.862519     | 0.054977            | 0.0284% | 70.49739 |

Here is the result of orginal paper.

\begin{array}{c|ccc}
\hline \hline \text { OASIS dataset } & \text { Avg. Dice }(28) \uparrow & \mathcal{D}_\psi(\mathrm{x}) \leq 0\left(r^{\mathcal{D}}\right) \downarrow & \mathcal{D}_\psi(\mathrm{x}) \leq 0\left(s^{\mathcal{D}}\right) \downarrow \\
\hline \text { SYMNet [25] } & 0.743 \pm 0.113 & 0.026 \% & - \\
\text { SyN [1] } & 0.729 \pm 0.109 & 0.026 \% & 0.005 \\
\text { NiftyReg [2] } & 0.775 \pm 0.087 & 0.102 \% & 1395.988 \\
\text { Log-Demons [3] } & 0.764 \pm 0.098 & 0.121 \% & 84.904 \\
\text { NODEO (ours } \left.\lambda_1=2.5\right) & 0.778 \pm 0.026 & 0.030 \% & 34.183 \\
\text { NODEO (ours } \lambda_1=2 \text { ) } & \mathbf{0 . 7 7 9} \pm \mathbf{0 . 0 2 6} & 0.030 \% & 61.105 \\
\hline \text { CANDI dataset } & \text { Avg. Dice }(28) \uparrow & \mathcal{D}_\psi(\mathrm{x}) \leq 0\left(r^{\mathcal{D}}\right) \downarrow & \mathcal{D}_\psi(\mathrm{x}) \leq 0\left(s^{\mathcal{D}}\right) \downarrow \\
\hline \text { SYMNet [25] } & 0.778 \pm 0.091 & 1.4 \times 10^{-4 \%} & 1.043 \\
\text { SyN [1] } & 0.739 \pm 0.102 & 0.018 \% & 0.012 \\
\text { NiftyReg [2] } & 0.775 \pm 0.088 & 0.101 \% & 1395.987 \\
\text { Log-Demons [3] } & 0.786 \pm 0.094 & 0.071 & 49.274 \\
\text { NODEO (ours } \left.\lambda_1=2.5\right) & 0.801 \pm 0.011 & 7.5 \times 10^{-8 \%} & 1.574 \\
\text { NODEO (ours } \left.\lambda_1=2\right) & \mathbf{0 . 8 0 2} \pm \mathbf{0 . 0 1 1} & 1.8 \times 10^{-7 \%} \% & 4.341 \\
\hline \text { CANDI dataset } & \text { Avg. Dice }(32) \uparrow & \mathcal{D}_\psi(\mathrm{x}) \leq 0\left(r^{\mathcal{D}}\right) \downarrow & \mathcal{D}_\psi(\mathrm{x}) \leq 0\left(s^{\mathcal{D}}\right) \downarrow \\
\hline \text { SYMNet [25] } & 0.736 \pm 0.015 & 1.4 \times 10^{-4 \%} \% & 1.043 \\
\text { SyN [1] } & 0.713 \pm 0.177 & 0.018 \% & 0.012 \\
\text { NiftyReg [2] } & 0.748 \pm 0.160 & 0.101 \% & 1395.987 \\
\text { Log-Demons [3] } & 0.744 \pm 0.160 & 0.071 & 49.274 \\
\text { NODEO (ours } \left.\lambda_1=2.5\right) & \mathbf{0 . 7 6 0} \pm \mathbf{0 . 0 1 1} & 7.5 \times 10^{-8 \%} \% & 1.574 \\
\text { NODEO (ours } \left.\lambda_1=2\right) & \mathbf{0 . 7 6 0} \pm \mathbf{0 . 0 1 1} & 1.8 \times 10^{-7} \% & 4.341 \\
\hline
\end{array}

# Applying code on new data

## Characteristics of orignal dataset

### Image content

The original paper applies the code on OASIS and CANDI datasets. These datasets are similar in the way that both contain images of brain scans. To collect the OASIS datasets, brain scans of adults are used whereas the subjects of the CANDI dataset are children and adolescents

### Image format

Both datasets used in the original paper contain images in a .nii.gz format. Images in this format contain pixel information in 3 dimensions, which represents multiple slices of a brain scan. In addition 3D image info, the datasets contain segments of the subjects, which represent a slice of the side view of the scanned images.

### Image dimensionality

The images used for the results in the table presented in the paper are all 3-dimansional

## Required properties of input

### Image content

For the neural network described in the paper to be able to properly learn, the fixed and moving images given as the input should be of the same subject. It must be possible for the moving image to deform into the fixed image, which doesn't make sense if the fixed and moving images given to the model as input are not from the same subject.

### Image format

The data used in the paper is given in .nii.gz format. However, this is not a requirement for the model to work. The code in its original format can take any 3D image information with a corresponding segmentation slice. The code provided on github converts .nii.gz images to a 3D array of the pixel values. However this step can be modified to convert images of another data type to a 3D array, and then apply the code from that point on.

### Image dimensionality

In addition to being able to take 3D images of a different format, the code can be modified to be able to process 2D images. How this can be done is described in the previous section on reproduction of the original results. With this modification, the code can be applied on any image type

To summarize, either 3D or 2D images of any format can be used as input for the code, the only hard requirement is that the fixed and moving images are of the same subject, taken at different times.

### New dataset

In order to compare with new input, the FIRE (Fundus Image REgistration) dataset taken from [4].

One combination of inputs can be seen below, for this subject A01 was taken:

<center>

Fixed                                    |  Moving
:-------------------------:|:-------------------------:
![](https://i.imgur.com/7wu3aZO.jpg =300x300) |  ![](https://i.imgur.com/tCLK5Ce.jpg =300x300)
</center>
### Modifications made to the original code

Since the images in the dataset are 2D, the 2D adaptation of the code is used, which has been discussed in the reproduction segment. In addition to that, the following setings were used:
<center>
    
| Parameter          | Value           | 
| :-----------------: | :-----------------: | 
| Image size   | 512x512   | 
| Epoches   | 50   | 
| Learning rate   | 0.005   | 
| Kernel type   | Averaging kernel   | 

</center>

### Results

The code outputs the following images for the warped moving image and the grid visualization of the deformation field:

<center>Output from running the code

Warped moving                                    |  Grid visualization of deformation field
:-------------------------:|:-------------------------:
![](https://i.imgur.com/sJ6CuUi.png =300x250)   |  ![](https://i.imgur.com/sajKUXM.png =300x300)
</center>

Using the beforementioned label indexes, the mean dice were calculated for an average of 15 structures, the results of which can be seen below for 13 pairs of input images (in the dataset they are labeled A01 to A014):

<center>

| Dice avg.          | Dice std.           | Neg. J ratio    |
| :-----------------: | :-----------------: | :-------------: |
| 0.692387574220678   | 0.11240498671185131   | 0.0   |
| 0.6642838965724711  | 0.18842665050725657   | 0.0   |
| 0.690506568746174   | 0.2000820351598212   | 0.0   |
| 0.6629531615910564  | 0.17187152578295337   | 0.0   |
| 0.7406926138147159  | 0.15772506039841624   | 0.0   |
| 0.795548832526095   | 0.12641128263148094   | 0.0   |
| 0.8884103250603481   | 0.09727527291491649   | 0.0   |
| 0.873301915530903   | 0.05216006748560099   | 0.0   |
| 0.8158854356576679   | 0.06494718990986122   | 0.0   |
| 0.7899100455687188   | 0.0864340539586151   | 0.0   |
| 0.7826216180477299   | 0.1270870510016491   | 0.0   |
| 0.6243754567234394   | 0.10244470618228722   | 0.0   |
| 0.5809893135464423   | 0.17781713506801744   | 0.0   |

</center>


## Hyperparameter analysis

A simple ANOVA analysis was performed on some hyperparameters, in order to find out a bit more on which hyperparameter has the most effect on the quality of the output of the code. In order to determine the quality of the output we used two metrics: dice average and ratio of negative Jacobian determinant values.

### Selected hyperparameters

In order to do the ANOVA analysis, we chose the following hyperparameters and values:

| Kernel          | Learning rate           | # of epoches    |
| :-----------------: | :-----------------: | :-------------: |
| Averaging Kernel   | 0.001   | 5   |
| Gaussian Kernel  | 0.005   | 10   |
| -   | 0.01   | 20   |

### ANOVA analysis

ANalysis Of VAriance (ANOVA) can be performed when you want to determine which parameter has the most significant impact on the outcome of some process. In addition to individual effect we also analyse how their interactions influence the outcome. Using ANOVA formulas and statistical analysis we compute the F-Value which gives information about the significance of each variable and interaction, and we compare this to the critical value obtained from the standardized F-table. More information regarding ANOVA analysis can be found in [5].

It is common convention to rename parameters to a letter, to have a nicer visual representation of the table. We mapped the parameters as follows:

<center>

| Parameter          | Mapping           | 
| :-----------------: | :-----------------: | 
| Kernel type   | A   | 
| Leraning rate  | B   | 
| # of epoches   | C   | 

</center>
    
### Results

In the created ANOVA tables we look for computed F-values that have a value larger than the critical value ni the standard F-table.

<center>

Anova analysis for dice average as metric 
    
|   var |       ss | dof |       ms |   F_comp | F_table  |
|------:|---------:|----:|---------:|---------:|----------|
|     A | 0.000034 |   1 | 0.000034 | 0.015168 | 4.413873 |
|     B | 0.005058 |   2 | 0.002529 | 1.135638 | 3.554557 |
|     C | 0.025219 |   2 | 0.012610 | 5.661957 | 3.554557 |
|    AB | 0.007939 |   2 | 0.003970 | 1.782415 | 3.554557 |
|    AC | 0.001489 |   2 | 0.000745 | 0.334331 | 3.554557 |
|    BC | 0.003861 |   4 | 0.000965 | 0.433366 | 2.927744 |
|   ABC | 0.005454 |   4 | 0.001364 | 0.612274 | 2.927744 |
| Error | 0.040087 |  18 | 0.002227 | 1.000000 |      - |
    
</center>

Here we can see the ANOVA table for dice avg as metric. We can see that most of the computed F values are smaller than those in the F-table, with the # of epoches being the exception. The next most significant parameters are the interaction between kernel type and learning rate and the learning rate.

It would be naive to say that based on this table that the learning rate for example does not have a significant effect on the outcome. We can only conclude that for the analyzed range the parameters and interactions with a lower F-value have an insignificant effect, the expectation is that for a wider range the effect would be larger. 



<center>

Anova analysis for ratio of negative values in the Jacobian Determinant 
    
|   var |       ss | dof |       ms |   F_comp | F_table  |
|------:|---------:|----:|---------:|---------:|----------|
|     A | 0.000057 |   1 | 0.000057 | 0.259242 | 4.413873 |
|     B | 0.000256 |   2 | 0.000128 | 0.581635 | 3.554557 |
|     C | 0.000579 |   2 | 0.000289 | 1.316811 | 3.554557 |
|    AB | 0.000548 |   2 | 0.000274 | 1.246425 | 3.554557 |
|    AC | 0.000138 |   2 | 0.000069 | 0.314941 | 3.554557 |
|    BC | 0.000504 |   4 | 0.000126 | 0.573858 | 2.927744 |
|   ABC | 0.000948 |   4 | 0.000237 | 1.078730 | 2.927744 |
| Error | 0.003956 |  18 | 0.000220 | 1.000000 |      - |
</center>

Based on the second ANOVA table we can conclude that for the selected range of hyperparameter values none of them have a significant effect.

# Conclusion
In this blog we have reproduced results from the paper by Wu et al. We have shown that and how using the code provided on github the promised results can be achieved. In addition to that we have shown that the code can be applied on the FIRE dataset, which is also made for DIR applications. An ANOVA analysis was done on several hyperparameters which showed that for the selected range of values only the number of epoches had a significant impact on the results.




 


# Reference
[1] Wu Y, Jiahao T Z, Wang J, et al. Nodeo: A neural ordinary differential equation based optimization framework for deformable image registration[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 20804-20813.

[2] Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. Journal of cognitive neuroscience, 19(9), 1498-1507.

[3] https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md

[4] Larxel, 2020. Fundis Image REgistration dataset. Taken from: https://www.kaggle.com/datasets/andrewmvd/fundus-image-registration

[5] E. Howell, Jun 2022. ANOVA Test Simply Explained. Taken from: https://towardsdatascience.com/anova-test-simply-explained-c94e4620ec6f