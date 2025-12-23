import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import librosa
import hydra
import random
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from models.common import LoadPretrainedBase
from utils.config import register_omegaconf_resolvers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

register_omegaconf_resolvers()


import warnings 


warnings.filterwarnings("ignore")

    



def load_and_process_audio(audio_path: str, target_sr: int) -> torch.Tensor:

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, orig_sr = torchaudio.load(str(path))

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim > 2:
        waveform = waveform.reshape(-1)

    if target_sr and int(target_sr) != int(orig_sr):
        waveform_np = waveform.cpu().numpy()
        resampled_np = librosa.resample(
            waveform_np, 
            orig_sr=int(orig_sr), 
            target_sr=int(target_sr)
        )
        waveform = torch.from_numpy(resampled_np)
        
    return waveform





def main():

    # tf32 of H200 may lead bugs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = []
    # load Config
    
    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]
    seed = 42
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    # Resolve 'exp_dir' and checkpoint path
    if "exp_dir" in config:
        exp_dir = Path(config["exp_dir"])
        use_best = config.get("use_best", True)
        ckpt_dir = exp_dir / "checkpoints"
        
        # Sort checkpoints by time or name
        checkpoints = sorted(ckpt_dir.iterdir())
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            
        if use_best:
            ckpt_path = checkpoints[0] / "model.safetensors"
        else:
            ckpt_path = checkpoints[-1] / "model.safetensors"
            
    elif "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir
    else:
        raise ValueError("Config must contain 'exp_dir' or 'ckpt_dir'")

    logger.info(f"Checkpoint path: {ckpt_path}")

    #  Load Model

    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    
    # Load weights
    train_sd = load_file(ckpt_path)
    model.load_pretrained(train_sd)
    model = model.to(device)
    model.eval()

    scheduler = getattr(
        noise_schedulers,
        config["noise_scheduler"]["type"],
    ).from_pretrained(
        config["noise_scheduler"]["name"],
        subfolder="scheduler",
    )


    wav_dir = Path(config["wav_dir"])
    output_subdir = config.get("output_subdir", "inference")
    gen_audio_dir = wav_dir / output_subdir
    gen_audio_dir.mkdir(parents=True, exist_ok=True)

    target_sr = exp_config.get("sample_rate", 24000)

    
    waveform = load_and_process_audio(config["audio_path"], target_sr)
    


    batch = {
        "audio_id": [Path(config["audio_path"]).stem], 
        "content": [
        {
            "audio": waveform,      
            "caption": config["caption"] 
        }
    ],
        "task": ["audio_editing"]                
    }

    def to_device(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: to_device(v) for k, v in data.items()}
        return data

    batch["content"][0]["audio"] = batch["content"][0]["audio"].to(device)

    logger.info("Starting inference...")
    with torch.no_grad():
        kwargs = config.get("infer_args", {}).copy()
        kwargs.update(batch)
        waveform = model.inference(
            scheduler=scheduler,
            **kwargs,
        )
        audio_id = batch["audio_id"][0]
        out_filename = f"{audio_id}.wav"
        out_path = gen_audio_dir / out_filename
        
        logger.info(f"Saving generated audio to: {out_path}")
        

        output_audio = waveform[0, 0].cpu().numpy()
        
        sf.write(
            out_path,
            output_audio,
            samplerate=target_sr,
        )

    logger.info("Inference finished successfully.")

if __name__ == "__main__":
    main()