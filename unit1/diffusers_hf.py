# from huggingface_hub import login

import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from diffusers import DDPMPipeline


# login(token='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', add_to_git_credential=True)


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def get_dataset(image_size=32):
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )
    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    return dataset


def get_model(image_size=32):
    # Create a model
    model = UNet2DModel(sample_size=image_size,  # the target image resolution
                        in_channels=3,  # the number of input channels, 3 for RGB images
                        out_channels=3,  # the number of output channels
                        layers_per_block=2,  # how many ResNet layers to use per UNet block
                        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
                        down_block_types=(
                            "DownBlock2D",  # a regular ResNet downsampling block
                            "DownBlock2D",
                            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                            "AttnDownBlock2D",
                        ),
                        up_block_types=(
                            "AttnUpBlock2D",
                            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                            "UpBlock2D",
                            "UpBlock2D",  # a regular ResNet upsampling block
                        ),
                    )
    return model


def train_model(image_size=32, batch_size=16, epochs=30, timesteps=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('setting dataset...')
    dataset = get_dataset(image_size=32)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('setting model...')
    model = get_model(image_size=image_size)
    model.to(device)

    # Set the noise scheduler
    print('setting scheduler...')
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")

    # Training loop
    print('setting optimizer...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    losses = []
    print('start training...')
    for epoch in range(epochs):
        print(f'training epoch {epoch}...')
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
    print('saving model...')
    model.save_pretrained('weights')
    """Plotting the loss, we see that the model rapidly improves initially and then continues to get better at a slower rate (which is more obvious if we use a log scale as shown on the right):"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()
    plt.savefig('./train_loss.png')


def save_pipeline(image_size=32, timesteps=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('setting model...')
    model = get_model(image_size=image_size)
    model.to(device)
    # Set the noise scheduler
    print('setting scheduler...')
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")

    pipeline = DDPMPipeline(unet=model.from_pretrained('weights'), scheduler=noise_scheduler)
    pipeline.save_pretrained("pipeline")


def test_model(pipeline_dir='pipeline', batch=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = DDPMPipeline.from_pretrained(pipeline_dir).to(device)
    return pipeline(batch_size=batch).images


if __name__ == '__main__':
    # train_model(image_size=32, batch_size=16, epochs=30)

    # save_pipeline(image_size=32)

    images = test_model(batch=1)
    images[0].save('result.jpg', 'jpeg')
