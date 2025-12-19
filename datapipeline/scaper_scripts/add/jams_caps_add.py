import os
import json
from pathlib import Path
import argparse
from typing import Dict, List
import numpy as np
import pandas as pd
import jams
from tqdm import tqdm
from scaper.core import _sample_trunc_norm
import json


### 用于读取csv字段
def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# ===== S3 读取辅助（petrel_client 可选） =====
from pathlib import Path
import io



def parse_onset(onset_str):
    """
    输入: 
        onset_str = "a_man_talks__0.320-1.160_2.520-3.080--water_pours__1.240-9.720"
    输出:
        {
            "a_man_talks": [(0.32, 1.16), (2.52, 3.08)],
            "water_pours": [(1.24, 9.72)]
        }
    """
    result = {}
    if onset_str.strip() == "":
        return result  # 没有标注

    events = onset_str.split("--")
    for event in events:
        if "__" not in event:
            continue
        label, segments = event.split("__")
        seg_list = []
        for seg in segments.split("_"):
            start, end = seg.split("-")
            seg_list.append((float(start), float(end)))
        result[label] = seg_list
    return result



def load_csv_for_background(file_path):
    """
    从背景CSV文件中加载 location 和 caption 信息作为标签。
    返回:
        bg_file_list: 文件路径列表 (Path对象)
        bg_aid_to_duration: {file_path: duration}
        bg_caption_to_files: {caption: [file1, file2, ...]}
        bg_caption_to_cnt: {caption: count}
    """
    data = load_jsonl(file_path)
    bg_file_list = []
    bg_aid_to_duration = {}
    bg_caption_to_files = {}
    bg_caption_to_cnt = {}

    for item in data:
        file_path_str = item["location"]
        file_path = Path(file_path_str)
        caption = item.get("captions", "unknown_background_sound") # 使用caption作为标签
        
        # 实际项目中应通过读取音频文件获取真实时长
        duration = 10.0  # 默认认为是10秒，实际应从音频文件获取
        # 如果CSV中没有提供时长信息，并且也不依赖onset计算，则统一设为10秒
        # 否则，如果可以通过librosa等库读取，可以在这里添加读取逻辑

        bg_aid_to_duration[file_path] = duration
        bg_file_list.append(file_path)

        bg_caption_to_files.setdefault(caption, []).append(file_path)
        bg_caption_to_cnt[caption] = bg_caption_to_cnt.get(caption, 0) + 1

    return bg_file_list, bg_aid_to_duration, bg_caption_to_files, bg_caption_to_cnt




## 原始的逻辑类似于add背景，先确定前景事件，然后add背景，总体时间不会改变


def select_valid_sound_from_list(sound_files: List,
                                 aid_to_duration: Dict,
                                 random_state: np.random.RandomState):
    sound_file = random_state.choice(sound_files, 1)[0]
    # aid_to_duration 存储的键是 Path 对象，所以直接用 Path(sound_file) 作为键
    duration = aid_to_duration.get(sound_file, 0.0) 
    
    if duration == 0.0:
        # Fallback to a fixed duration if not found (should ideally not happen with correct data loading)
        print(f"Warning: Duration not found for {sound_file}, using default 10.0s")
        duration = 10.0

    while duration < 0.01: # 确保时长不是0
        sound_file = random_state.choice(sound_files, 1)[0]
        duration = aid_to_duration.get(sound_file, 0.0)
        if duration == 0.0:
            duration = 10.0 # fallback
    return sound_file, duration


def select_snr(mean=10.0,
               std=5.0,
               min=-5.0,
               max=25.0,
               random_state=None):
    snr = _sample_trunc_norm(mean, std, min, max, random_state)
    return snr


def generate_one_sound(sound_files: List,
                       aid_to_duration: Dict,
                       max_num_occurrence: int,
                       min_single_event_duration: float,
                       max_single_event_duration: float,
                       cur_duration: float,
                       init_onset: float,
                       sound_type: str,
                       time_sensitive: bool,
                       id_sensitive: bool,
                       max_clip_duration: float,
                       min_interval: float,
                       max_interval: float,
                       sound_info_list: List,
                       repeat_single: bool,
                       times_control: bool,
                       random_state: np.random.RandomState,
                       loud_threshold: float = 15.0,
                       low_threshold: float = 2.5,
                       continuation_prob: float = 1.0,
                       info_type: str = 'timestamp',):

    """
    Args:
        cur_duration (float): the current maximum offset of all sounds
        sound_info_list (list): output list for jams, will be modified in-place
    """
    
    sound_file = None # 初始化 sound_file
    source_time = 0.0 # 默认从头开始
    duration = 0.0 # 初始化时长


    sound_file, duration = select_valid_sound_from_list(sound_files,
                                                        aid_to_duration,
                                                        random_state)
    
    # TODO maximum tolerant single event duration
    if duration > 10.0:
        trunc_duration = random_state.uniform(5.0, 10.0)
        source_time = random_state.uniform(0, duration - trunc_duration)
        duration = trunc_duration
    else:
        source_time = 0.0

    # randomly sample a loudness
    snr = select_snr(random_state=random_state)

    if snr > loud_threshold:
        loudness = "loud"
    elif snr < low_threshold:
        loudness = "low"
    else:
        loudness = None

    # determine when to start: overlap / continuation
    continuation = True if random_state.rand() <= continuation_prob else False
    if continuation:
        interval = random_state.uniform(min_interval, max_interval)
        cur_duration_event = cur_duration + min(interval, max_clip_duration - cur_duration)
    else:
        # overlap with added sounds
        if cur_duration > duration:
            cur_duration_event = random_state.uniform(0, cur_duration - duration)
        else:
            cur_duration_event = random_state.uniform(0, init_onset)

    metadata = {
        "sound_type": sound_type,
        "start": round(cur_duration_event, 3)
    }
    if loudness is not None:
        metadata["loudness"] = loudness

    if not time_sensitive and id_sensitive and not repeat_single:
        metadata_list = []
    
    cur_sound_infos = []

    if random_state.random() < 0.5: # 这里的逻辑看起来有点随意，可以根据实际需求调整
        max_interval = min(
            max_clip_duration / max_num_occurrence - min_single_event_duration,
            max_interval
        )

    event_end = None

    orig_sound_files = sound_files.copy()

    for occur_i in range(max_num_occurrence):
        
        if cur_duration_event >= max_clip_duration:
            break

        if occur_i > 0 and not repeat_single: # 如果不是第一次出现，且不允许重复使用同一文件
            if len(sound_files) > 1:
                sound_files.remove(sound_file) # 移除已使用的文件
            else:
                sound_files = orig_sound_files.copy() # 如果只剩一个文件，则重置列表

            # 重新选择文件和时长
            if not sound_files: # 如果没有可用文件了，跳出
                break

            sound_file, duration = select_valid_sound_from_list(sound_files,
                                                                aid_to_duration,
                                                                random_state)
            # 如果使用onset，重新确定source_time和duration
        
            if duration > max_single_event_duration:
                trunc_duration = random_state.uniform(min_single_event_duration,
                                                        max_single_event_duration)
                source_time = random_state.uniform(0, duration - trunc_duration)
                duration = trunc_duration
            else:
                source_time = 0.0
            # TODO determine whether to resample snr in multiple files
            # snr = select_snr(random_state=random_state)

        occur_i_duration = min(duration, max_clip_duration - cur_duration_event)

        # if duration left for the current sound is too short, just stop, to avoid unnatural stop
        if occur_i_duration < duration and occur_i_duration < min_single_event_duration:
            break

        sound_info = {
            "time": cur_duration_event,
            "duration": occur_i_duration,
            "value": {
                "label": sound_type,
                "source_file": sound_file.__str__(),
                "source_time": source_time,
                "event_time": cur_duration_event,
                "event_duration": occur_i_duration,
                "snr": snr,
                "role": "foreground",
                "pitch_shift": None,
                "time_stretch": None
            }
        }
        cur_duration_event += occur_i_duration
        event_end = cur_duration_event
        sound_info_list.append(sound_info)
        cur_sound_infos.append(sound_info)
        
        if not time_sensitive and id_sensitive and not repeat_single:
            cur_metadata = metadata.copy()
            cur_metadata["id"] = occur_i + 1
            cur_metadata["start"] = round(sound_info["time"], 3)
            cur_metadata["end"] = round(event_end, 3)
            metadata_list.append(cur_metadata)

        if cur_duration_event >= max_clip_duration:
            break
        
        interval = random_state.uniform(min_interval, max_interval)
        cur_duration_event += min(interval, max_clip_duration - cur_duration_event)


    if len(cur_sound_infos) == 0:
        metadata = None
    else:
        if time_sensitive:
            if times_control:
                if id_sensitive:
                    metadata.update({
                        "end": round(event_end, 3),
                        "times": len(cur_sound_infos),
                    })
                    if len(cur_sound_infos) > 1:
                        metadata["id"] = "single" if repeat_single else "multiple"
                else:
                    metadata.update({
                        "end": round(event_end, 3),
                        "times": len(cur_sound_infos)
                    })
            else:
                metadata.update({
                    "end": round(event_end, 3),
                })
        else:
            if id_sensitive and not repeat_single:
                # different ids (e.g., multiple speakers)
                metadata = metadata_list
            else:
                metadata.update({
                    "end": round(event_end, 3)
                })

    cur_duration = max(cur_duration, cur_duration_event)
    return cur_duration, metadata


def generate(args):
    max_event_occurrence = args.max_event_occurrence
    add_bg_prob = args.add_bg_prob
    max_distinct_identity = args.max_distinct_identity
    max_single_event_duration = args.max_single_event_duration
    min_single_event_duration = args.min_single_event_duration
    repeat_single_prob = args.repeat_single_prob
    mean_init_onset = args.mean_init_onset
    min_init_onset = args.min_init_onset
    max_init_onset = args.max_init_onset
    clap_score_filter = args.clap_score_filter
    uniform_sampling_events = args.uniform_sampling_events
    fg_label_counts = {}
    bg_label_counts = {}

    output_dir = Path(args.output_jams_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output_dir = output_dir / "raw"
    edit_output_dir = output_dir / "edit"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    edit_output_dir.mkdir(parents=True, exist_ok=True)

    # 加载背景音数据：优先从CSV文件加载
    bg_files = []
    bg_aid_to_duration = {}
    bg_sound_type_to_files = {} # 这里的键会是 caption
    bg_sound_type_to_cnt = {}

    if args.bg_csv_file:
        # print(f"Loading background sounds from CSV: {args.bg_csv_file}")
        bg_files, bg_aid_to_duration, bg_sound_type_to_files, bg_sound_type_to_cnt = load_csv_for_background(args.bg_csv_file)
        # bg_sound_types 将是 caption 列表；将会有90000个caption啊
        bg_sound_types = sorted(list(bg_sound_type_to_files.keys()))
    else:
        # print(f"Loading background sounds from directory: {args.sound_type_dir} (for background if no CSV)")
        # 如果没有提供CSV，背景音仍然从 sound_type_dir 获取（原始逻辑）
        # 这里需要重新构建 bg_sound_type_to_files, bg_aid_to_duration 等，因为它们可能还没被填充
        # 或者在下面前景音加载的循环中填充
        pass # 在下面的循环中处理

    sound_type_to_files = {} # 用于前景音
    sound_type_to_cnt = {} # 用于前景音
    sound_types = [] # 用于前景音类别列表
    aid_to_duration = {} # 用于前景音时长信息

    fg_segment, bg_segment = 0, 0


    # 遍历 sound_type_dir 加载前景音数据
    for sound_type_path in Path(args.sound_type_dir).iterdir():
        sound_type = sound_type_path.stem
        sound_type_to_files[sound_type] = []
        for sound_file_path in Path(sound_type_path).iterdir():
            ## 确保文件名是预期的格式，可以安全地进行字符串切片
            # if len(sound_file_path.stem) < 11 or "__" not in sound_file_path.stem[11:]:
            #     print(f"Warning: Skipping malformed filename for foreground: {sound_file_path.name}")
            #     continue
            
            try:
                # 获取 clap_score
                clap_score_str_part = sound_file_path.stem[11:].split("__")[1]
                clap_score = round(float(clap_score_str_part), 3)

                # 获取时长
                duration_str_part = sound_file_path.stem[11:].split("__")[2]
                start_str, end_str = duration_str_part.split("_")
                duration = round(float(end_str) - float(start_str), 3)
            except (ValueError, IndexError) as e:
                print(f"Error parsing metadata from filename {sound_file_path.name}: {e}. Skipping.")
                continue

            if clap_score < clap_score_filter: continue


            ## 这是原本的加载逻辑
            # if len(sound_file_path.stem) < 11 or "__" not in sound_file_path.stem[11:]:
            #     print(f"Warning: Skipping malformed filename for foreground: {sound_file_path.name}")
            #     continue
            
            # try:
            #     # 获取 clap_score
            #     clap_score_str_part = sound_file_path.stem[11:].split("__")[1]
            #     clap_score = round(float(clap_score_str_part), 3)

            #     # 获取时长
            #     duration_str_part = sound_file_path.stem[11:].split("__")[2]
            #     start_str, end_str = duration_str_part.split("_")
            #     duration = round(float(end_str) - float(start_str), 3)
            # except (ValueError, IndexError) as e:
            #     print(f"Error parsing metadata from filename {sound_file_path.name}: {e}. Skipping.")
            #     continue

            if clap_score < clap_score_filter: continue
            
            # 前景音筛选逻辑
            if duration >= 0.5 and duration <= 6 :
                sound_type_to_files[sound_type].append(sound_file_path)    
                aid_to_duration[sound_file_path] = duration # 存储Path对象作为键
                fg_segment += 1

            # 如果背景音不是从CSV加载，则在这里也处理背景音数据;还是会从sound_type种读取
            if not args.bg_csv_file and duration >= 5:
                if sound_type not in bg_sound_type_to_files:
                    bg_sound_type_to_files[sound_type] = []
                bg_sound_type_to_files[sound_type].append(sound_file_path)
                bg_sound_type_to_cnt[sound_type] = bg_sound_type_to_cnt.get(sound_type, 0) + 1
                bg_files.append(sound_file_path)
                bg_aid_to_duration[sound_file_path] = duration 
                bg_segment += 1

        if len(sound_type_to_files[sound_type]) > 0:
            sound_types.append(sound_type)
            sound_type_to_cnt[sound_type] = len(sound_type_to_files[sound_type])
        else:
            del sound_type_to_files[sound_type]
    
    print(f"Total foreground sound type: {len(sound_type_to_files)}, foreground segments: {fg_segment}")

    # sorted_data = dict(sorted(sound_type_to_cnt.items(), key=lambda item: item[0], reverse=True))
    #print("前景音类别分布:", sorted_data, "\n")

    cnts = [sound_type_to_cnt[t] for t in sound_types]

    # 如果背景音不是从CSV加载，重新确定 bg_sound_types 列表
    if not args.bg_csv_file:
        bg_sound_types = sorted(list(bg_sound_type_to_files.keys()))
    
    bg_type_counts = [bg_sound_type_to_cnt[t] for t in bg_sound_types]
    # print(f"Total background sound types: {len(bg_sound_type_to_files)}, background segments: {bg_segment}")
    # print("背景音类型分布:", dict(sorted(bg_sound_type_to_cnt.items(), key=lambda item: item[0], reverse=False)))


    random_state = np.random.RandomState(args.seed)

    metas = {}

    num_events = 1 
    
    generated_num = 0
    total_occurrence = 0
    event_to_num = {idx:0 for idx in range(1, 6)} 
    with tqdm(total=args.syn_number) as pbar:
        while generated_num < args.syn_number:
            sound_info_list: List[Dict] = []
            metadata: List[Dict] = []

            init_onset = _sample_trunc_norm(mean_init_onset,
                                            0.5,
                                            min_init_onset,
                                            max_init_onset,
                                            random_state)
            cur_duration = init_onset

            # 采样一个前景事件类型
            available_fg_sound_types = [t for t in sound_types]
            
            if not available_fg_sound_types:
                print("Error: No available foreground sound types. Check sound_type_dir and filters.")
                break

            if uniform_sampling_events:
                fg_sound_type = random_state.choice(available_fg_sound_types)
            else:
                probs = np.array([cnts[sound_types.index(t)] for t in available_fg_sound_types])
                probs = probs / probs.sum()
                fg_sound_type = random_state.choice(available_fg_sound_types, p=probs)
            
            fg_sound_files = sound_type_to_files[fg_sound_type].copy()
            fg_label_counts[fg_sound_type] = fg_label_counts.get(fg_sound_type, 0) + 1

           
            id_sensitive_fg = True
            times_control = True
            # 每次都是在最大值中随机选一个
            max_num_occurrence = random_state.randint(1, max_event_occurrence + 1)
            
            repeat_single = random_state.random() < repeat_single_prob
            if not repeat_single:
                max_num_occurrence = random_state.randint(1, max_distinct_identity + 1)

            # 调用 generate_one_sound 生成单个前景事件
            cur_duration, metadata_i = generate_one_sound(
                sound_files=fg_sound_files,
                aid_to_duration=aid_to_duration, # 前景音使用自己的 aid_to_duration
                max_num_occurrence=max_num_occurrence,
                max_single_event_duration=max_single_event_duration,
                min_single_event_duration=min_single_event_duration,
                cur_duration=cur_duration,
                time_sensitive=True,
                id_sensitive=id_sensitive_fg,
                init_onset=init_onset,
                sound_type=fg_sound_type,
                max_clip_duration=args.max_duration,
                min_interval=args.min_interval,
                max_interval=args.max_interval,
                sound_info_list=sound_info_list,
                repeat_single=repeat_single,
                times_control=times_control,
                random_state=random_state,
                loud_threshold=args.loud_threshold,
                low_threshold=args.low_threshold,
                info_type = args.info_type,
            )

            if metadata_i is not None:
                if isinstance(metadata_i, dict):
                    metadata.append(metadata_i)
                elif isinstance(metadata_i, list) and len(metadata_i) == 1:
                    metadata.extend(metadata_i)
            
            if not metadata:
                continue

            # 选择背景音
            if not bg_sound_types:
                print("Error: No available background sound types. Check bg_csv_file or sound_type_dir and filters.")
                generated_num += 1
                pbar.update()
                continue
            
            # 排除与前景音类型相同的背景音
            available_bg_types_for_selection = [t for t in bg_sound_types if t != fg_sound_type]
            if not available_bg_types_for_selection:
                # 如果所有背景音类型都与前景音类型相同，则重新允许选择前景音类型作为背景音
                available_bg_types_for_selection = bg_sound_types
                print(f"Warning: All background types are same as foreground '{fg_sound_type}'. Allowing foreground type as background.")

            bg_selected_type = random_state.choice(available_bg_types_for_selection)

            # 从选定的背景音类型中选择一个文件
            bg_paths_for_type = bg_sound_type_to_files[bg_selected_type]
            if not bg_paths_for_type:
                print(f"Warning: No files found for background type '{bg_selected_type}'. Skipping.")
                generated_num += 1
                pbar.update()
                continue
            
            # 背景音不需要根据前景音时长进行裁剪，直接选择文件
            bg_path = random_state.choice(bg_paths_for_type)
            bg_duration = bg_aid_to_duration.get(bg_path, args.max_duration) # 使用存储的原始时长，或者默认值
            
            bg_info = {
                "time": 0.0,
                "value": {
                    "label": bg_selected_type, # 使用 caption 作为标签
                    "source_file": bg_path.__str__(),
                    "event_time": 0.0, # 背景音从0开始
                    "event_duration": min(bg_duration, cur_duration), # 背景音的时长不应超过最终剪辑时长
                    "snr": 0, 
                    "role": "background",
                    "pitch_shift": None,
                    "time_stretch": None
                },
                "confidence": 1.0
            }
            bg_label_counts[bg_selected_type] = bg_label_counts.get(bg_selected_type, 0) + 1 # 统计背景音标签

            bg_event_time = bg_duration
            bg_source_time = 0
            bg_info["duration"] = bg_event_time
            bg_info["value"].update({
                "source_time": bg_source_time,
                "event_duration": bg_event_time
            })

            ## add jams
            cur_duration=bg_duration
            jam = jams.JAMS()
            jam.file_metadata.duration = cur_duration
            ann = jams.Annotation(namespace="scaper", time=0, duration=cur_duration,
                                    sandbox={
                                        "scaper": {
                                            "duration": cur_duration,
                                            "original_duration": cur_duration,
                                            "fg_path": args.fg_path,
                                            "bg_path": args.bg_path,
                                            "protected_labels": [],
                                            "sr": args.sample_rate,
                                            "ref_db": args.ref_db,
                                            "n_channels": 1,
                                            "fade_in_len": 0.1,
                                            "fade_out_len": 0.1,
                                            "reverb": args.reverb,
                                            "disable_sox_warnings": True
                                        }
                                    })

            

            # 添加背景音到JAMS
            # 这个是添加前景的代码欸
            if random_state.random() < add_bg_prob:
                ann.append(**bg_info)



            jam.annotations.append(ann)
            raw_path = raw_output_dir / f"syn_{generated_num}.jams"
            jam.save(str(raw_path))

            
            
            ### 添加前景到混合jams

            for sound_info in sound_info_list:
                ann.append(**sound_info)
            jam2 = jams.JAMS()
            jam2.file_metadata.duration = cur_duration
            jam2.annotations.append(ann)

            edit_path = edit_output_dir / f"syn_{generated_num}.jams"
            jam2.save(str(edit_path))
            

            meta = {
                "pair_id": generated_num,
                "duration": round(cur_duration, 3),
                "raw_jams_file": str(raw_path),
                "edited_jams_file": str(edit_path),
                "foreground_event": metadata,
                "background_info": {
                    "type": bg_selected_type,
                    "file": str(bg_path),
                    "snr": 0
                } 
            }
            metas[f"pair_{generated_num}"] = meta
            generated_num += 1
            total_occurrence += len(sound_info_list) 
            event_to_num[len(meta["foreground_event"])] += 1 
            
            pbar.update()

    print(event_to_num, total_occurrence)
    json.dump(metas, open(args.output_meta, "w"), indent=2)

    print("\n--- 生成完成后的统计 ---")
    print("前景事件标签数量：")
    for label, count in fg_label_counts.items():
        print(f"  {label}: {count}")

    # print("\n背景事件标签数量：")
    # for label, count in bg_label_counts.items():
    #     print(f"  {label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound_type_dir", type=str, required=True,
                        help="Directory containing foreground sound types categorized in subdirectories.")
    parser.add_argument("--bg_csv_file", type=str, default=None,
                        help="CSV file containing background sound locations and caption information. If provided, background will be loaded from here.")
    parser.add_argument("--max_duration", type=float, required=True)
    parser.add_argument("--syn_number", "-n", type=int, required=True)
    parser.add_argument("--clap_score_filter", type=float, required=True)
    parser.add_argument("--output_jams_dir", type=str, required=True)
    parser.add_argument("--output_meta", type=str, required=True)
    parser.add_argument("--info_type", type=str, default='timestamp', 
                        choices=["timestamp", "duration", "interval", "ordering", "frequency"])

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_num_events", type=int, default=1) 
    parser.add_argument("--max_num_events", type=int, default=1) 

    parser.add_argument("--max_event_occurrence", type=int, default=3) 
    parser.add_argument("--min_interval", type=float, default=0.0)
    parser.add_argument("--max_interval", type=float, default=1.0)
    parser.add_argument("--max_distinct_identity", type=int, default=1) 
    parser.add_argument("--times_desc_prob", type=float, default=0.8) # Not directly relevant now
    parser.add_argument("--add_bg_prob", type=float, default=1)
    
    parser.add_argument("--uniform_sampling_events", default=True, action="store_true")

    parser.add_argument("--repeat_single_prob", type=float, default=1.0) 

    # duration setting
    parser.add_argument("--mean_init_onset", type=float, default=0.05)
    parser.add_argument("--min_init_onset", type=float, default=0.0)
    parser.add_argument("--max_init_onset", type=float, default=0.2)
    parser.add_argument("--max_single_event_duration",
                        type=float,
                        default=7.0,
                        help="Maximum duration for a single foreground sound")
    parser.add_argument("--min_single_event_duration",
                        type=float,
                        default=2.0,
                        help="Minimum duration for a single foreground sound (when cropping)")
    
    # loudness setting
    parser.add_argument("--loud_threshold", type=float, default=12.5)
    parser.add_argument("--low_threshold", type=float, default=2.5)

    # scaper argument
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--ref_db", type=float, default=-40.0)
    parser.add_argument("--reverb", type=float, default=None)
    parser.add_argument("--fg_path", type=str, default="/mnt/petrelfs/taoye/workspace/data/audiocaps/audio_clap")
    parser.add_argument("--bg_path", type=str, default="/mnt/petrelfs/taoye/workspace/data/audiocaps/audio_clap")


    args = parser.parse_args()

    job_name = Path(args.output_meta).stem
    synth_meta_fpath = Path(args.output_meta).with_name(job_name + "_synth_hp.json")
    if not synth_meta_fpath.parent.exists():
        synth_meta_fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(synth_meta_fpath, "w") as writer:
        json.dump(vars(args), writer, indent=4)

    generate(args)