# Chest-X-rays-Segmentation-with-CycleGAN

The network will be trained on unpaired images of chest X-rays and lung masks to learn a forward mapping (X-rays to Mask) and a reverse mapping (Mask to X-rays). The network consists of:
* 2 Generators with U-Net architecture, and 6 residual blocks. One generator learns the forward mapping, and the other learns the reverse mapping.
* 2 Discriminators classifies whether the generated images are real or fake. <br/>

The network can be applied when there is no paired images of chest X-rays and lung mask, since the network can create a lung mask that corresponds to a chest X-ray.


.
 * [tree-md](./tree-md)
 * [dir2](./dir2)
   * [file21.ext](./dir2/file21.ext)
   * [file22.ext](./dir2/file22.ext)
   * [file23.ext](./dir2/file23.ext)
 * [dir1](./dir1)
   * [file11.ext](./dir1/file11.ext)
   * [file12.ext](./dir1/file12.ext)
 * [file_in_root.ext](./file_in_root.ext)
 * [README.md](./README.md)
 * [dir3](./dir3)
