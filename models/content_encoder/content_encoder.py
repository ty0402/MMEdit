from typing import Any
import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        text_encoder: nn.Module = None,
        llm_encoder: nn.Module = None,
        video_encoder: nn.Module = None,
        midi_encoder: nn.Module = None,
        phoneme_encoder: nn.Module = None,
        pitch_encoder: nn.Module = None,
        audio_encoder: nn.Module = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_encoder = text_encoder
        self.midi_encoder = midi_encoder
        self.phoneme_encoder = phoneme_encoder
        self.pitch_encoder = pitch_encoder
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder

    def encode_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_content_output = []
        batch_content_mask = []
        batch_la_content_output = []
        batch_la_content_output_mask = []
        zero_la_content = torch.zeros(1, 1, self.embed_dim, device=device)
            
        for i,(content, task) in enumerate(zip(batch_content, batch_task)):
            if task == "audio_editing":
                raw_waveform = torch.as_tensor(content["audio"]).float()
                waveform_with_batch_dim = raw_waveform.unsqueeze(0).to(device)
                waveform_lengths = torch.as_tensor([raw_waveform.shape[0]])
                
                # Note: text encoder actually is audiollm encoder, encode both waveform and caption 
                content_output_dict = self.text_encoder(
                    [content["caption"]], waveform_with_batch_dim
                )
                audio_dict = {
                        "waveform": waveform_with_batch_dim,
                        "waveform_lengths": waveform_lengths
                    }
                audio_output_dict = self.audio_encoder(**audio_dict)
                la_content_output_dict = {
                    "output": audio_output_dict["output"],
                    "mask": audio_output_dict["mask"]
                }

            batch_content_output.append(content_output_dict["output"][0])
            batch_content_mask.append(content_output_dict["mask"][0])
            batch_la_content_output.append(la_content_output_dict["output"][0])
            batch_la_content_output_mask.append(
                la_content_output_dict.get("mask", zero_la_content)[0]
            )

        batch_content_output = nn.utils.rnn.pad_sequence(
            batch_content_output, batch_first=True, padding_value=0
        )
        batch_content_mask = nn.utils.rnn.pad_sequence(
            batch_content_mask, batch_first=True, padding_value=False
        )
        batch_la_content_output = nn.utils.rnn.pad_sequence(
            batch_la_content_output, batch_first=True, padding_value=0
        )

        batch_la_content_output_mask = nn.utils.rnn.pad_sequence(
            batch_la_content_output_mask, batch_first=True, padding_value=False
        )
        return {
            "content": batch_content_output ,
            "content_mask": batch_content_mask,
            "length_aligned_content": batch_la_content_output,
            "time_aligned_content_mask": batch_la_content_output_mask
        }



class BatchedContentEncoder(ContentEncoder):
    def encode_content(
        self, batch_content: list[dict], batch_task: list[str],
        device: str | torch.device
    ):
        assert all(task == "audio_editing" for task in batch_task), \
            "BatchedContentEncoder now are only support audio_editing"

        zero_la_content = torch.zeros(1, 1, self.embed_dim, device=device)

        captions = []
        waveforms = []
        waveform_lengths = []
        for content in batch_content:
            raw_waveform = torch.as_tensor(content["audio"]).float().to(device)
            captions.append(content["caption"])
            waveforms.append(raw_waveform)  
            waveform_lengths.append(raw_waveform.shape[0])

        content_output_dict = self.text_encoder(
            captions, waveforms
        )

        batch_la_content_output = []
        batch_la_content_output_mask = []
        for i in range(len(batch_content)):
            audio_dict = {
                "waveform": waveforms[i].unsqueeze(0),
                "waveform_lengths": torch.as_tensor([waveform_lengths[i]], device=device)
            }
            audio_output_dict = self.audio_encoder(**audio_dict)
            batch_la_content_output.append(audio_output_dict["output"][0])
            batch_la_content_output_mask.append(audio_output_dict["mask"][0])

        # pad audio_encoder 
        batch_la_content_output = nn.utils.rnn.pad_sequence(
            batch_la_content_output, batch_first=True, padding_value=0
        )
        batch_la_content_output_mask = nn.utils.rnn.pad_sequence(
            batch_la_content_output_mask, batch_first=True, padding_value=False
        )

        return {
            "content": content_output_dict["output"],      
            "content_mask": content_output_dict["mask"],   
            "length_aligned_content": batch_la_content_output,
            "time_aligned_content_mask": batch_la_content_output_mask
        }
