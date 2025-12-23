from pathlib import Path

import soundfile as sf
import torch
import hydra
import re
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase
from utils.general import sanitize_filename

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    if "exp_dir" in config:
        use_best = config.get("use_best", True)
        exp_dir = Path(config["exp_dir"])
        if use_best:  # use best ckpt
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                    )[0] / "model.safetensors"
        else:  # use last ckpt
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                    )[-1] / "model.safetensors"
    elif "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir

    print(f'\n ckpt path: {ckpt_path}\n ')

    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    train_sd = load_file(ckpt_path)
    model.load_pretrained(train_sd)

    model = model.to(device)
    if "sampler" in config["test_dataloader"]:
        data_source = hydra.utils.instantiate(
            config["test_dataloader"]["dataset"], _convert_="all"
        )
        sampler = hydra.utils.instantiate(
            config["test_dataloader"]["sampler"],
            data_source=data_source,
            _convert_="all"
        )
        test_dataloader = hydra.utils.instantiate(
            config["test_dataloader"], sampler=sampler, _convert_="all"
        )
    else:
        test_dataloader = hydra.utils.instantiate(
            config["test_dataloader"], _convert_="all"
        )

    model.eval()

    scheduler = getattr(
        noise_schedulers,
        config["noise_scheduler"]["type"],
    ).from_pretrained(
        config["noise_scheduler"]["name"],
        subfolder="scheduler",
    )

    audio_output_dir = exp_dir / config["wav_dir"]
    audio_output_dir.mkdir(parents=True, exist_ok=True)
   

    output_subdir = config.get("output_subdir", "cfg_0")
    gen_audio_dir = audio_output_dir / output_subdir
    gen_audio_dir.mkdir(parents=True, exist_ok=True)
    gen_audio_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            for key in list(batch.keys()):
                data = batch[key]
                if isinstance(data, torch.Tensor):
                    batch[key] = data.to(device)

            kwargs = config["infer_args"].copy()
            kwargs.update(batch)
            waveform = model.inference(
                scheduler=scheduler,
                **kwargs,
            )
            audio_id = batch["audio_id"][0]

            if isinstance(batch["audio_id"][0], str):
                out_file: str = batch["audio_id"][0]
            else:
                out_file: str = str(batch["audio_id"][0])

            if not out_file.endswith(".wav"):
                out_file = f"{out_file}.wav"

            # raw_output_dir = audio_output_dir / "raw"
            # raw_output_dir.mkdir(parents=True, exist_ok=True)

            # raw_audio = batch["content"][0]["audio"].cpu().numpy()

            # raw_audio = raw_audio.reshape(-1, 1)  
            # sf.write(
            #     raw_output_dir / out_file,
            #     raw_audio,
            #     samplerate=exp_config["sample_rate"],
            # )

            sf.write(
                gen_audio_dir / out_file,
                waveform[0, 0].cpu().numpy(),
                samplerate=exp_config["sample_rate"],
            )


if __name__ == "__main__":
    main()
