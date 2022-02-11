# Chest-X-rays-Segmentation-with-CycleGAN

![output](https://user-images.githubusercontent.com/63922382/153581632-515cebb6-f157-4264-bff7-0840d92830c8.png)

The network will be trained on unpaired images of chest X-rays and lung masks to learn a forward mapping (X-rays to Mask) and a reverse mapping (Mask to X-rays). The network consists of:
* 2 Generators with U-Net architecture, and 6 residual blocks. One generator learns the forward mapping, and the other learns the reverse mapping.
* 2 Discriminators classifies whether the generated images are real or fake. <br/>

The network can be applied when there is no paired images of chest X-rays and lung mask, since the network can create a lung mask that corresponds to a chest X-ray.

## Dependencies
Python 3 <br/>
Numpy <br/>
PyTorch <br/>

## Prepare Training Data
Prepare chest X-ray images and lung mask images. Have chest X-rays and lung masks in 2 seperate folders.

## Train The Model
```
python main.py

## Specify the dataset path (required)
--mask_root                                     # The path of folder of lung mask images
--x_ray_root                                    # The path folder of chest X-ray images

## Preprocess 
--image_size                                    # Size of the image after being resized, default (128, 128)
--in_channels                                   # The number of channels of the image, defaul 1

## Model Parameters
--num_residual_blocks                           # The number of residual blocks for the generator, default 6
--lambda_cycle                                  # Weight of the cycle loss, default 10.0
--lambda-identity                               # Weight of the identity mapping loss, default 0.5

## Train Parameters 
--num_epochs                                    # Number of epochs with the initial learning rate, default 50
--num_epochs_decay                              # Number of epochs to linearly decaying the learning rate to zero, default 0
--batch_size                                    # Size of training batch, default 1
--learning_rate                                 # The initial learning rate of Adam optimizer, default 2e-4
--beta1                                         # Momentum term of Adam optimizer, default 0.5
--g_accumulation_steps                          # Gradient accumulation steps for generators, default 1
--d_accumulation_steps                          # Gradient accumulation steps for discriminators, default 1

## Validation Parameters
--validate_image_path                           # The path of the image for validation, default None
--validate_frequency                            # Frequency of validation, default 1
```

After training, a file called `checkpoint.pt` will be automatically saved. The file contains parameters of generators, discriminators, optimizers, and schedulers.

## Generate Images
```
python test.py

## Required
--test_image_folder                             # Specify the folder of test images
--generated_image_folder                        # Specify the folder that will save generated images

## Chose generator
--generator_type                                # ["mask", "x_ray"]. "mask" will generate lung masks. "x_ray" will generate chest X-rays.
```
