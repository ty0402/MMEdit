from typing import Sequence
import random
from typing import Any

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.schedulers as noise_schedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor

from models.autoencoder.autoencoder_base import AutoEncoderBase
from models.content_encoder.content_encoder import ContentEncoder
from models.content_adapter import ContentAdapterBase, ContentEncoderAdapterMixin
import soundfile as sf
from models.common import (
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
)
from utils.torch_utilities import (
    create_alignment_path, create_mask_from_length, loss_with_mask,
    trim_or_pad_length
)


class DiffusionMixin:
    def __init__(
        self,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        cfg_drop_ratio: float = 0.2
    ) -> None:
        self.noise_scheduler_name = noise_scheduler_name
        self.snr_gamma = snr_gamma
        self.classifier_free_guidance = cfg_drop_ratio > 0.0
        self.cfg_drop_ratio = cfg_drop_ratio
        self.noise_scheduler = noise_schedulers.DDPMScheduler.from_pretrained(
            self.noise_scheduler_name, subfolder="scheduler"
        )

    def compute_snr(self, timesteps) -> torch.Tensor:
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod)**0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device
                                                    )[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[...,
                                                                          None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma)**2
        return snr

    def get_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        training: bool = True
    ) -> torch.Tensor:
        if training:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size, ),
                device=device
            )
        else:
            # validation on half of the total timesteps
            timesteps = (self.noise_scheduler.config.num_train_timesteps //
                         2) * torch.ones((batch_size, ),
                                         dtype=torch.int64,
                                         device=device)

        timesteps = timesteps.long()
        return timesteps

    def get_input_target_and_timesteps(
        self,
        latent: torch.Tensor,
        training: bool,
    ):
        batch_size = latent.shape[0]
        device = latent.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        timesteps = self.get_timesteps(batch_size, device, training=training)
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        target = self.get_target(latent, noise, timesteps)
        return noisy_latent, target, timesteps

    def get_target(
        self, latent: torch.Tensor, noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the target for loss depending on the prediction type
        """
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                latent, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )
        return target

    def loss_with_snr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
        reduce: bool = True
    ) -> torch.Tensor:
        if self.snr_gamma is None:
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask, reduce=reduce)
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py#L1006
            snr = self.compute_snr(timesteps)
            mse_loss_weights = torch.stack(
                [
                    snr,
                    self.snr_gamma * torch.ones_like(timesteps),
                ],
                dim=1,
            ).min(dim=1)[0]
            # division by (snr + 1) does not work well, not clear about the reason
            mse_loss_weights = mse_loss_weights / snr
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss_with_mask(loss, mask, reduce=False) * mse_loss_weights
            if reduce:
                loss = loss.mean()
        return loss

    def rescale_cfg(
        self, pred_cond: torch.Tensor, pred_cfg: torch.Tensor,
        guidance_rescale: float
    ):
        """
        Rescale `pred_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_cond = pred_cond.std(
            dim=list(range(1, pred_cond.ndim)), keepdim=True
        )
        std_cfg = pred_cfg.std(dim=list(range(1, pred_cfg.ndim)), keepdim=True)

        pred_rescaled = pred_cfg * (std_cond / std_cfg)
        pred_cfg = guidance_rescale * pred_rescaled + (
            1 - guidance_rescale
        ) * pred_cfg
        return pred_cfg


class SingleTaskCrossAttentionAudioDiffusion(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    DiffusionMixin, ContentEncoderAdapterMixin
):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        content_dim: int,
        noise_scheduler_name: str = "stabilityai/stable-diffusion-2-1",
        snr_gamma: float = None,
        cfg_drop_ratio: float = 0.2,
    ):
        nn.Module.__init__(self)
        DiffusionMixin.__init__(
            self, noise_scheduler_name, snr_gamma, cfg_drop_ratio
        )
        ContentEncoderAdapterMixin.__init__(
            self, content_encoder=content_encoder
        )
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        if hasattr(self.content_encoder, "audio_encoder"):
            self.content_encoder.audio_encoder.model = self.autoencoder

        self.backbone = backbone
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self, content: list[Any],  task: list[str],
        waveform: torch.Tensor, waveform_lengths: torch.Tensor, **kwargs
    ):
        device = self.dummy_param.device

        self.autoencoder.eval()
        self.content_encoder.eval()
        with torch.no_grad():
            
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths,pad_latent_len=500
            )

        with torch.no_grad():
            content_dict = self.content_encoder.encode_content(content, task, device)
        context, context_mask = content_dict["content"], content_dict[
            "content_mask"]
        time_aligned_content = content_dict["length_aligned_content"]
        time_aligned_content_mask = content_dict[
            "time_aligned_content_mask"
        ]
        latent_mask = time_aligned_content_mask.to(device)

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                context[mask_indices] = 0
                # dont mask!
                # time_aligned_content[mask_indices] = 0

        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent, self.training
        )

        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            time_aligned_context=time_aligned_content,
            context=context,
            x_mask=latent_mask,
            context_mask=context_mask
        )
        
        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        loss = self.loss_with_snr(pred, target, timesteps, latent_mask)

        return loss

    def prepare_latent(
        self, batch_size: int, scheduler: SchedulerMixin,
        latent_shape: Sequence[int], dtype: torch.dtype, device: str
    ):
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=dtype
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latent = latent * scheduler.init_noise_sigma
        return latent

    def iterative_denoise(
        self,
        latent: torch.Tensor,
        scheduler: SchedulerMixin,
        verbose: bool,
        cfg: bool,
        cfg_scale: float,
        cfg_rescale: float,
        backbone_input: dict,
    ):
        timesteps = scheduler.timesteps
        num_steps = len(timesteps)
        num_warmup_steps = len(timesteps) - num_steps * scheduler.order
        progress_bar = tqdm(range(num_steps), disable=not verbose)

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            if cfg:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent
            latent_input = scheduler.scale_model_input(latent_input, timestep)
            # print(latent_input.shape)
            noise_pred = self.backbone(
                x=latent_input, timesteps=timestep, **backbone_input
            )

            # perform guidance
            if cfg:
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_content - noise_pred_uncond
                )
                if cfg_rescale != 0.0:
                    noise_pred = self.rescale_cfg(
                        noise_pred_content, noise_pred, cfg_rescale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latent = scheduler.step(noise_pred, timestep, latent).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                           (i + 1) % scheduler.order == 0):
                progress_bar.update(1)

        progress_bar.close()

        return latent

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        task: list[str],
        scheduler: SchedulerMixin,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        disable_progress: bool = True,
        mask_time_aligned_content: bool = True,  # 新增参数
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(content)


        content_dict = self.content_encoder.encode_content(content, task, device)
        

        context, context_mask = content_dict["content"], content_dict[
            "content_mask"]
        time_aligned_content = content_dict["length_aligned_content"]
        time_aligned_content_mask = content_dict[
            "time_aligned_content_mask"
        ]



        B, T, C = time_aligned_content.shape
        latent_shape = (C, T)      # 128, 500
        latent_mask=time_aligned_content_mask.to(device)

        

        if classifier_free_guidance:


            if mask_time_aligned_content:
                uncond_time_aligned_content = torch.zeros_like(time_aligned_content)
            else:
                uncond_time_aligned_content = time_aligned_content.detach().clone()

            uncond_context = torch.zeros_like(context)
            uncond_context_mask = context_mask.detach().clone()
            time_aligned_content = torch.cat([
                uncond_time_aligned_content, time_aligned_content
            ])
            context = torch.cat([uncond_context, context])
            context_mask = torch.cat([uncond_context_mask, context_mask])
            latent_mask = torch.cat([
                latent_mask, latent_mask.detach().clone()
            ])

        scheduler.set_timesteps(num_steps, device=device)
        
        latent = self.prepare_latent(
            batch_size, scheduler, latent_shape, context.dtype, device
        )

        latent = self.iterative_denoise(
            latent=latent,
            scheduler=scheduler,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            cfg_rescale=guidance_rescale,
            backbone_input={
                "x_mask": latent_mask,
                "context": context,
                "context_mask": context_mask,
                "time_aligned_context": time_aligned_content,
            }
        )
        waveform = self.autoencoder.decode(latent,latent_mask)

        return waveform


