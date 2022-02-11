# Chest-X-rays-Segmentation-with-CycleGAN

The network will be trained on unpaired images of chest X-rays and lung masks to learn a forward mapping (X-rays to Mask) and a reverse mapping (Mask to X-rays). The network consists of:
* 2 Generators with U-Net architecture, and 6 residual blocks. One generator learns the forward mapping, and the other learns the reverse mapping.
* 2 Discriminators classifies whether the generated images are real or fake. <br/>

The network can be applied when there is no paired images of chest X-rays and lung mask, since the network can create a lung mask that corresponds to a chest X-ray.

## Dependencies
Python 3 <br/>
Numpy <br/>
PyTorch <br/>

## Preparing data
Prepare chest X-ray images and lung mask images. Have chest X-rays and lung masks in 2 seperate folders.

## Train the model
```
python main.py

## Specify the dataset path (required)
--mask_root                                     # The path of folder of lung mask images
--x_ray_root                                    # The path folder of chest X-ray images

## Preprocess 
--image_size (128, 128)                         # Size of the image after being resized
--in_channels 1                                 # The number of channels of the image

## Model Parameters
--num_residual_blocks 6                         # The number of residual blocks for the generator
--lambda_cycle 10.0                             # Weight of the cycle loss
--lambda-identity 0.5                           # Weight of the identity mapping loss

## Train Parameters 
--num_epochs                                    # 
--num_epochs_decay                              #
--batch_size                                    #
--learning_rate                                 #
--beta1                                         #
--g_accumulation_steps                          #
--d_accumulation_steps                          #

## Validation Parameters
--validate_image_path                           #
--validate_frequency                            #
```




