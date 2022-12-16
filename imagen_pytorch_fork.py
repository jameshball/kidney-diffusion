"""
MIT License

Copyright (c) 2022 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Further modified by jameshball
"""

import torch
from imagen_pytorch import Imagen
from tqdm.auto import tqdm


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ImagenFork(Imagen):
    @torch.no_grad()
    def p_sample_loop(
            self,
            unet,
            shape,
            *,
            noise_scheduler,
            lowres_cond_img=None,
            lowres_noise_times=None,
            text_embeds=None,
            text_mask=None,
            cond_images=None,
            inpaint_images=None,
            inpaint_masks=None,
            inpaint_resample_times=5,
            init_images=None,
            skip_steps=None,
            cond_scale=1,
            pred_objective='noise',
            dynamic_threshold=True,
            use_tqdm=True
    ):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device=device)

        # for initialization with an image or video

        if exists(init_images):
            img += init_images

        # keep track of x0, for self conditioning

        x_start = None

        # prepare inpainting

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-1])
            inpaint_masks = self.resize_to(inpaint_masks, shape[-1]).bool()

        # time

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device=device)

        # whether to skip any steps

        skip_steps = default(skip_steps, 0)
        timesteps = timesteps[skip_steps:]

        for times, times_next in tqdm(timesteps, desc='sampling loop time step', total=len(timesteps),
                                      disable=not use_tqdm):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, *_ = noise_scheduler.q_sample(inpaint_images, t=times)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next=times_next,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    cond_images=cond_images,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold
                )

                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )

        img.clamp_(-1., 1.)

        # final inpainting

        if has_inpainting:
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img