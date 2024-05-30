import torch
import numpy as np
from tqdm import tqdm

HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = 512 // 8
LATENT_WIDTH = 512 // 8

def generate(
        prompt,
        unconditional_prompt,
        inp_img=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inf_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None):

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError()

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        # creating seeded generator
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # classifier free guidance using tokenizer (clip)
        if do_cfg:
            # prompt to tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, sequence_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # takes the prompts for each i in batch and returns emb (batch_size, sequence_len, dim)
            cond_context = clip(cond_tokens)

            uncond_token = tokenizer.batch_encode_plus([unconditional_prompt], padding="max_length", max_length=77).input_ids
            uncond_token = torch.tensor(uncond_token, dtype=torch.long, device=device)
            uncond_token = clip(uncond_token)

            # (2, 77, 768) 2 prompts
            context = torch.cat([cond_tokens, uncond_token])

        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768) 1 prompt
            context = clip(tokens)

        to_idle(clip)



        sampler= DDPMSampler(generator)
        sampler.set_inference_steps(n_inf_steps)

        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if inp_img:
            encoder = models["encoder"]
            encoder.to(device)
            input_image_tensor = inp_img.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

