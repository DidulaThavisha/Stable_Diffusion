import torch
import numpy as np
from tqdm import tqdm 
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 512//8
LATENT_HEIGHT = 512//8


def rescale(x, in_range, out_range, clamp=False):
    temp = (x - in_range[0]) * (out_range[1] - out_range[0]) / (in_range[1] - in_range[0]) + out_range[0]
    if clamp:
        temp = torch.clamp(temp, out_range[0], out_range[1])
    return temp

def get_time_embedding(timestep):

    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)              # (160,)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]                # (1, 160)         
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)                                  # (1, 320))

@torch.no_grad()
def generate(
        prompt: str, 
        unconditional_prompt: str,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None
        ):
    if not(0 <= strength <= 1):
        raise ValueError("Strength must be between 0 and 1")
    
    if idle_device:
        to_idle_device = lambda x: x.to(idle_device)
    else:
        to_idle_device = lambda x: x

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generate.seed()
    
    clip = models['clip']
    clip.to(device)

    # Classifier Free Guided Diffusion
    if do_cfg:
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens, device=device, dtype=torch.long)    # (B, T)
        cond_text = clip(cond_tokens)                                               # (B, T, C)

        uncond_tokens = tokenizer.batch_encode_plus([unconditional_prompt], padding="max_length", max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, device=device, dtype=torch.long)    # (B, T)
        uncond_text = clip(uncond_tokens)                                               # (B, T, C)

        context = torch.cat([cond_text, uncond_text])                               # (2B, T, C) = (2B , 77, 768)

    else:
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens, device=device, dtype=torch.long)
        context = clip(cond_tokens)

    to_idle_device(clip)

    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_steps(n_inference_steps)
    else:
        raise ValueError(f"Sampler {sampler_name} not supported")
    
    latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

    if input_image:
        encoder = models['encoder']
        encoder.to(device)

        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)      # (H, W, 3)

        input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
        input_image_tensor = input_image_tensor.permute(2,0,1)                          # (3, H, W)
        input_image_tensor = input_image_tensor.unsqueeze(0)                            # (1, 3, H, W)

        encoder_noise = torch.randn(latent_shape, generator=generator, device=device)

        latents = encoder(input_image_tensor, encoder_noise)                            # (1, 4, H//8, W//8) 

        sampler.set_strength(strengt=strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        to_idle_device(encoder)

    else:
        # If we are doing teext-based generation, we start with a random noise latent tensor, N(0,I)
        latents = torch.randn(latent_shape, generator=generator, device=device)
    
    # Generate the image
    diffusion = models['diffusion']
    diffusion.to(device)


    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        # (1, 320)
        time_embedding = get_time_embedding(timestep).to(device)                        # (1, 320)
        model_input = latents
        if do_cfg:
            model_input = model_input.repeat(2, 1, 1, 1)                                # (2, 4, H//8, W//8)

        model_output = diffusion(model_input, context, time_embedding)                   # (2, 4, H//8, W//8)    

        if do_cfg:
            output_cond, output_uncond = torch.chunk(model_output, 2, dim=0)             # (1, 4, H//8, W//8) each
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

        latents = sampler.step(timestep, latents, model_output)   

    to_idle_device(diffusion)

    decoder = models['decoder'] 
    decoder.to(device)

    image = decoder(latents)                                                            # (1, 3, H, W) 
    to_idle_device(decoder)

    image = rescale(image, (-1, 1), (0, 255), clamp=True)
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    image = image[0].astype(np.uint8)

    return image