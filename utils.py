# utils.py
import torch

def generate_digit_images(generator, digit, num_images=5, noise_dim=100):
    generator.eval()
    device = next(generator.parameters()).device
    noise = torch.randn(num_images, noise_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        generated_imgs = generator(noise, labels).cpu()
    return generated_imgs
