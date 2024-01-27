import os
import random
import numpy as np
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator

parser = ArgumentParser()
parser.add_argument(
    '--image_dir', default='/Downloads/CACD_VS/', help='The image directory')


@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('trained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(image_paths)
    # fig, axs = plt.subplots(2, max(nr_images, 2), figsize=(20, 10))
    # # Reshape axs to be 2D in case of a single image
    # axs = axs.reshape(2, -1)

    # random.shuffle(image_paths)
    # for i in range(nr_images):
    #     img = Image.open(image_paths[i]).convert('RGB')
    #     img = trans(img).unsqueeze(0)
    #     aged_face = model(img)
    #     aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
    #     axs[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
    #     axs[1, i].imshow(aged_face)
    # plt.show()
    fig, axs = plt.subplots(2, nr_images, figsize=(5 * nr_images, 10))  # Adjust the figsize dynamically based on nr_images
    if nr_images == 1:
        axs = np.expand_dims(axs, axis=-1)  # Reshape to 2D if only one image

    random.shuffle(image_paths)
    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        axs[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        axs[1, i].imshow(aged_face)
        axs[0, i].axis('off')  # Hide the axis for a cleaner look
        axs[1, i].axis('off')  # Hide the axis for a cleaner look

    plt.tight_layout()
    plt.show()
    plt.savefig("mygraph.png")


if __name__ == '__main__':
    main()
