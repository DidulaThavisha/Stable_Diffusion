import torch
import torch.nn as nn
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start = 0.00085, beta_end = 0.0120):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # [1, a0, a0*a1, a0*a1*a2, ...]
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(num_training_steps)[::-1].copy())
    
    def set_inference_steps(self, n_inference_steps = 50):
        self.n_inference_steps = n_inference_steps
        # 0, 19, 39, 59 ... 999
        step_ratio = self.num_training_steps // self.n_inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, self.n_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64))

    def add_noise(self, original_latents, timestep):
        mean = self.alphas_cumprod.to(original_latents.device, dtype=original_latents.dtype)
        timestep = timestep.to(original_latents.device)

        sqrt_alpha_t = self.alphas_cumprod[timestep]**0.5
        sqrt_alpha_t = sqrt_alpha_t.flatten()
        while len(sqrt_alpha_t.shape) < len(original_latents.shape):
            sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)

        sqrt_one_minus_alpha_t = (self.one - self.alphas_cumprod[timestep])**0.5
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.flatten()
        while len(sqrt_one_minus_alpha_t.shape) < len(original_latents.shape):
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.unsqueeze(-1)

        noise = torch.randn(original_latents.shape, generator=self.generator, device=original_latents.device, dtype=original_latents.dtype)
        noisy_latents = mean * original_latents + sqrt_one_minus_alpha_t * noise
        return noisy_latents
        
    def _get_previous_timestep(self, timestep):
        prev_t = timestep - (self.num_training_steps // self.n_inference_steps)
        return prev_t
    
    def _get_varience(self, timestep):  
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t/alpha_prod_t_prev

        variance = (1-alpha_prod_t_prev)/(1-alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def step(self, timestep, latents, model_output):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha = alpha_prod_t / alpha_prod_prev_t
        current_beta = 1 - current_alpha

        pred_original_sample = (latents - beta_prod_t**0.5 * model_output)/alpha_prod_t**0.5

        pred_original_sample_coeff = (alpha_prod_prev_t**0.5 * current_beta)/beta_prod_t
        current_sample_coeff = current_alpha**0.5 * beta_prod_prev_t/beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t>0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = self._get_varience(t)**0.5 * noise
        
        return pred_prev_sample + variance

    def set_strength(self, strength=1.0):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
