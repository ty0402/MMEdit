from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, Sequence
import json
import pickle
import random
from tqdm import tqdm
import numpy as np
from h5py import File
import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision
import soundfile as sf
import io


import librosa

def _is_s3_path(p: str | Path) -> bool:
    return isinstance(p, (str, Path)) and str(p).lower().startswith("s3://")


def read_jsonl_to_mapping(
    jsonl_file: str | Path, key_col: str, value_col: str
) -> dict[str, str]:
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            key = data[key_col]
            value = data[value_col]
            mapping[key] = value
    return mapping


def read_from_h5(key: str, h5_path: str, cache: dict[str, str] | None = None):
    if cache is None:
        with File(h5_path, "r") as reader:
            return reader[key][()]
    else:
        if h5_path not in cache:
            cache[h5_path] = File(h5_path, "r")
        return cache[h5_path][key][()]


@dataclass(kw_only=True)
class HDF5DatasetMixin:
    def __post_init__(self) -> None:
        self.h5_cache: dict[str, File] = {}

    def __del__(self) -> None:
        for h5_file in self.h5_cache.values():
            if h5_file:
                try:
                    h5_file.close()
                except:
                    pass


try:
    from petrel_client.client import Client
    _HAS_PETREL = True
    client = Client()
except ImportError:
    _HAS_PETREL = False



@dataclass(kw_only=True)
class AudioWaveformDataset(HDF5DatasetMixin):
    """
    can read s3 path if petrel_client is installed
    """

    target_sr: int | None = 24000
    use_h5_cache: bool = True


    silence_tokens: tuple[str, ...] = ("__SILENCE__", "silence", "SILENCE")
    silence_seconds: float = 10.0
    
    def __post_init__(self):
        super().__post_init__()
        self.h5_src_sr_map = {}

    def _make_silence(self, sr: int | None = None) -> torch.Tensor:
        """
        生成指定采样率下、长度为 silence_seconds 的一维静音音频。
        """
        sr = sr or self.target_sr
        num_samples = int(round(self.silence_seconds * sr))
        return torch.zeros(num_samples, dtype=torch.float32)

    # add load_waveform method from none
    def load_waveform(self, audio_id: str, audio_path: str):

        if (not audio_path) or (isinstance(audio_path, str) and audio_path.strip() == ""):
            return self._make_silence(24000)

        audio_path_str = str(audio_path)
        if audio_path_str in self.silence_tokens:
            return self._make_silence(24000)

        # 2) 
        if audio_path_str.endswith(".hdf5") or audio_path_str.endswith(".h5"):
            if _is_s3_path(audio_path_str):
                raise RuntimeError("not support")

            # on guizhou file system, using cached h5py.File will cause OOM error
            if self.use_h5_cache:
                waveform = read_from_h5(audio_id, audio_path_str, self.h5_cache)
            else:
                waveform = read_from_h5(audio_id, audio_path_str)

            if audio_path_str not in self.h5_src_sr_map:
                with File(audio_path_str, "r") as hf:
                    self.h5_src_sr_map[audio_path_str] = hf["sample_rate"][()]
            orig_sr = int(self.h5_src_sr_map[audio_path_str])
            waveform = torch.as_tensor(waveform, dtype=torch.float32).reshape(-1)

        else:
            if _is_s3_path(audio_path_str):
                data = client.get(audio_path_str)  # bytes
                audio_np, sr = sf.read(io.BytesIO(data), dtype="float32")  
                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)  
                waveform = torch.from_numpy(audio_np.copy())
                orig_sr = int(sr)
            else:
                waveform, orig_sr = torchaudio.load(audio_path_str)  
                if waveform.ndim == 2:
                    waveform = waveform.mean(0)  # [C, T] -> [T]
                else:
                    waveform = waveform.reshape(-1)

        # 3) resample

        waveform_np = waveform.cpu().numpy()

        if int(orig_sr) > 24000:
            if waveform_np.ndim == 1:
                waveform_np = librosa.resample(
                    waveform_np, orig_sr=int(orig_sr), target_sr=16000
                )
            else:
                tmp_channels = []
                for c in range(waveform_np.shape[0]):
                    resampled_c = librosa.resample(
                        waveform_np[c], orig_sr=int(orig_sr), target_sr=16000
                    )
                    tmp_channels.append(resampled_c)
                waveform_np = np.stack(tmp_channels, axis=0)
            orig_sr = 16000  
            waveform= torch.from_numpy(waveform_np).to(waveform.device)

        if self.target_sr and int(self.target_sr) != int(orig_sr):

            waveform_np = waveform.cpu().numpy()

            if waveform_np.ndim == 1:
                resampled_np = librosa.resample(
                    waveform_np, orig_sr=int(orig_sr), target_sr=int(self.target_sr)
                )
            else:
                resampled_channels = []
                for c in range(waveform_np.shape[0]):
                    resampled_c = librosa.resample(
                        waveform_np[c], orig_sr=int(orig_sr), target_sr=int(self.target_sr)
                    )
                    resampled_channels.append(resampled_c)
                resampled_np = np.stack(resampled_channels, axis=0)

            waveform = torch.from_numpy(resampled_np).to(waveform.device)

        return waveform


@dataclass
class AudioGenerationDataset(AudioWaveformDataset):

    content: str | Path
    audio: str | Path | None = None
    condition: str | Path | None = None
    id_col: str = "audio_id"
    id_col_in_content: str | None = None
    content_col: str = "content"
    id_col_in_audio: str | None = None
    audio_col: str = "audio"
    id_col_in_condition: str | None = None
    condition_col: str = "condition"

    def __post_init__(self, ):
        super().__post_init__()

        id_col_in_content = self.id_col_in_content or self.id_col
        self.id_to_content = read_jsonl_to_mapping(
            self.content, id_col_in_content, self.content_col
        )

        id_col_in_audio = self.id_col_in_audio or self.id_col
        if self.audio:
            self.id_to_audio = read_jsonl_to_mapping(
                self.audio, id_col_in_audio, self.audio_col
            )
        else:
            self.id_to_audio = None

        if self.condition:
            id_col_in_condition = self.id_col_in_condition or self.id_col
            self.id_to_condition = read_jsonl_to_mapping(
                self.condition, id_col_in_condition, self.condition_col
            )
        else:
            self.id_to_condition = None

        self.audio_ids = list(self.id_to_content.keys())

    @property
    @abstractmethod
    def task(self):
        ...

    def __len__(self) -> int:
        return len(self.audio_ids)

    @abstractmethod
    def load_condition(self, audio_id: str, condition_path: str) -> Any:
        ...

    @abstractmethod
    def load_content(self, audio_id: str, content_or_path: str) -> Any:
        ...

    @abstractmethod
    def load_duration(self, content: Any,
                      waveform: torch.Tensor) -> Sequence[float]:
        ...

    def load_content_waveform(self, audio_id: str) -> tuple[Any, torch.Tensor]:
        content_or_path = self.id_to_content[audio_id]
        content = self.load_content(audio_id, content_or_path)

        if self.id_to_audio:  # training, audio is the target
            audio_path = self.id_to_audio[audio_id]
            waveform = self.load_waveform(audio_id, audio_path)
        else:  # inference, only content is available
            waveform = None

        duration = self.load_duration(content, waveform)

        return content, waveform, duration

    def __getitem__(self, index) -> dict[str, Any]:
        audio_id = self.audio_ids[index]
        content, waveform, duration = self.load_content_waveform(audio_id)

        if self.id_to_condition:
            condition_path = self.id_to_condition[audio_id]
            condition = self.load_condition(audio_id, condition_path)
        else:
            condition = None

        return {
            "audio_id": audio_id,
            "content": content,
            "waveform": waveform,
            "condition": condition,
            "duration": duration,
            "task": self.task
        }




@dataclass(kw_only=True)
class AudioeditingDataset(AudioGenerationDataset):
    

    content: str | Path
    audio: str | Path | None = None
    use_h5_cache: bool = False
    downsampling_ratio: int | None
    id_col: str = "audio_id"
    content_col: str = "content"
    caption_col: str = "caption"
    audio_col: str = "audio"


    def __post_init__(self):
        super().__post_init__()

        # get content and caption from content.jsonl
        self.id_to_content = read_jsonl_to_mapping(
            self.content, self.id_col, self.content_col
        )
        self.id_to_caption = read_jsonl_to_mapping(
            self.content, self.id_col, self.caption_col
        )

        # get target audio from audio.jsonl
        if self.audio:
            self.id_to_audio = read_jsonl_to_mapping(
                self.audio, self.id_col, self.audio_col
            )
        else:
            self.id_to_audio = None
        
        self.audio_ids = list(self.id_to_content.keys())

    @property
    def task(self):
        return "audio_editing"

    def __len__(self) -> int:
        return len(self.audio_ids)

    def load_content(self, audio_id: str, content_or_path: str) -> Any:

        waveform = self.load_waveform(audio_id, content_or_path)

        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        return waveform
    

    def load_duration(self, content: Any, waveform: torch.Tensor) -> Sequence[float]:

        if content.dim() == 1:
            duration_time = content.size(0)//self.downsampling_ratio
        else:
            duration_time = content.size(1)//self.downsampling_ratio

        duration_value =   self.downsampling_ratio / self.target_sr 
        duration = np.full(duration_time, duration_value)
        return duration

    def __getitem__(self, index) -> dict[str, Any]:
        
        audio_id = self.audio_ids[index]


        raw_wave_path = self.id_to_content[audio_id]
        raw_waveform = self.load_content(audio_id, raw_wave_path)
        

        caption = self.id_to_caption[audio_id]

        edit_waveform = None
        if self.id_to_audio:
            audio_path = self.id_to_audio[audio_id]
            edit_waveform = self.load_waveform(audio_id, audio_path)

        

        return {
            "audio_id": audio_id,
            "content": {"audio": raw_waveform, "caption": caption},
            "waveform": edit_waveform,  
            "task": self.task,
        }
    
    def __del__(self):
        if hasattr(self, 'h5_cache'):
            for h5_file in self.h5_cache.values():
                h5_file.close()
        
    
class AudioGenConcatDataset(Dataset):
    def __init__(self, datasets: list[AudioGenerationDataset]):
        self.datasets = datasets
        self.lengths = np.array([len(d) for d in datasets])
        self.cum_sum_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cum_sum_lengths - 1, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum_lengths[dataset_idx - 1]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]


if __name__ == '__main__':

    from tqdm import tqdm


    dataset = AudioeditingDataset(
        content= "/mnt/petrelfs/taoye/workspace/x_to_audio_generation/toydata/add2/train/content.jsonl",
        audio= "/mnt/petrelfs/taoye/workspace/x_to_audio_generation/toydata/add2/train/audio.jsonl",
        condition= None,
        target_sr = 24000,
        use_h5_cache = False,
        downsampling_ratio=480
    )

    
    i = 1
    sample = dataset[i]
    print(f"Sample {i}:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value}")
        else:
            print(f"  {key}: {value}")

    from collate_function import PaddingCollate

    collate_fn = PaddingCollate(pad_keys=["waveform", "audio"])
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_fn, batch_size=4
    )
    for batch in dataloader:
        print("Batch from DataLoader:")
        print(batch["content"][0]["audio"].shape)
        print(batch["content"][1]["audio"].shape)
        print(batch["content"][2]["audio"].shape)
        print(batch["content"][3]["audio"].shape)

        for key, value in batch.items():
            if (key == "content"):
                print(value[0])
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape {value.shape}")
            else:
                print(f"  {key}: {value}")
        break  
