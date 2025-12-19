import json
import random

# ====== 相对时间分区定义 ======
DEFAULT_REGIONS = [
    ("begin",  0.00, 0.35),
    ("middle", 0.35, 0.65),
    ("end",    0.70, 1.00),
]

# ====== 相对时间判定 ======
def get_relative_position_strict(start, end, total_duration, regions=None, eps=1e-6):
    """
    只有当事件 [start,end] 完全落在某个相对分区内，才返回短语；否则返回 ""。
    """
    if regions is None:
        regions = DEFAULT_REGIONS

    for name, r0, r1 in regions:
        seg_s = r0 * total_duration
        seg_e = r1 * total_duration
        if start + eps >= seg_s and end - eps <= seg_e:
            if name == "begin":
                return random.choice(["at the beginning", "near the start"])
            elif name == "middle":
                return random.choice(["in the middle", "around the middle"])
            else:
                return random.choice(["towards the end", "near the end"])
    return ""


# ====== 其他描述组件 ======
def get_loudness_desc(loudness_val, include_loudness_prob=0.2):
    if not loudness_val or random.random() > include_loudness_prob:
        return ""
    v = loudness_val.lower()
    if v == "loud":
        return random.choice(["loud", "prominent"])
    elif v == "low":
        return random.choice(["quiet", "faint", "subtle"])
    return ""


def get_duration_desc(start, end):
    duration = end - start
    if duration < 1.5:
        return "brief"
    elif duration < 3.0:
        return "short"
    elif duration < 5.0:
        return " "
    else:
        return "long"


def get_frequency_desc(times):
    if not times or times == 1:
        return ""
    elif times == 2:
        return "twice"
    elif times == 3:
        return "three times"
    elif times <= 4:
        return random.choice(["several times", "a few times"])
    else:
        return random.choice(["repeatedly", "multiple times"])


# ====== 指令生成 ======
def generate_instruction(pair_data, template_weights=None):
    if template_weights is None:
        template_weights = {
            "simple":        0.05,
            "precise_time":  0.05,
            "relative_time": 0.6,
            "context":       0.05,
            "hybrid":        0.10,
            "duration":      0.10,
            "frequency":     0.05,
        }

    pair_id  = pair_data.get("pair_id")
    duration = float(pair_data.get("duration", 10.0))
    fg_list  = pair_data.get("foreground_event", [])
    bg_info  = pair_data.get("background_info", {})

    if not fg_list:
        return f"pair_{pair_id}: No instruction generated - missing foreground event data."

    fg = fg_list[0]  # 直接取第一个事件（因为你的数据就是单事件）
    sound_type = fg.get("sound_type", "sound").replace("_", " ")
    start = fg.get("start", 0.0)
    end   = fg.get("end", duration)
    times = fg.get("times", 1)
    loudness_val = fg.get("loudness", "")

    bg_caption = bg_info.get("type", "the audio").replace("_", " ")

    # 各种描述
    loudness_desc  = get_loudness_desc(loudness_val)
    relative_pos   = get_relative_position_strict(start, end, duration)
    duration_desc  = get_duration_desc(start, end)
    frequency_desc = get_frequency_desc(times)
    freq_text = f" {frequency_desc}" if frequency_desc else ""

    add_verb = random.choice(["Add", "Insert", "Incorporate", "Mix in"])

    # 模板选择
    categories = list(template_weights.keys())
    weights    = list(template_weights.values())
    selected_category = random.choices(categories, weights=weights, k=1)[0]

    if selected_category == "simple":
        instruction = f"{add_verb} {sound_type}."
    elif selected_category == "precise_time":
        instruction = f"{add_verb} {loudness_desc} {sound_type} sound from {start:.1f}s to {end:.1f}s."
    elif selected_category == "relative_time":
        if relative_pos:
            instruction = f"{add_verb} {loudness_desc} {sound_type} sound {relative_pos} {freq_text}."
        else:
            instruction = f"{add_verb} {sound_type} {freq_text}."
    elif selected_category == "context":
        instruction = f"Over the sound of {bg_caption}, {add_verb.lower()} {loudness_desc} {sound_type}."
    elif selected_category == "hybrid":
        instruction = f"In the audio of {bg_caption}, {add_verb.lower()} {loudness_desc} {sound_type}{freq_text}."
    elif selected_category == "duration":
        if relative_pos:
            instruction = f"{add_verb} {duration_desc} {sound_type} sound {relative_pos}."
        else:
            instruction = f"{add_verb} {duration_desc} {sound_type}."
    elif selected_category == "frequency":
        core = f"{add_verb} {sound_type} sound that occurs {frequency_desc or 'once'}"
        instruction = f"{core} {relative_pos}." if relative_pos else f"{core}."

    return f"pair_{pair_id}: {' '.join(instruction.split())}"


# ====== 主流程 ======
def process_json_file(json_file_path, output_file_path):
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    if not isinstance(data, dict):
        print("JSON format error: expected dict at top-level")
        return

    all_instructions = []
    for pair_data in data.values():
        if isinstance(pair_data, dict) and "pair_id" in pair_data:
            ins = generate_instruction(pair_data)
            all_instructions.append(ins)

    with open(output_file_path, "w") as f:
        for line in all_instructions:
            f.write(line + "\n")

    print(f"Generated {len(all_instructions)} instructions and saved to {output_file_path}")


# ====== 运行入口 ======
if __name__ == "__main__":
    json_file = "/mnt/petrelfs/taoye/workspace/editing/data/add/audiocaps/add_fore3/meta/add/audiocaps/add_fore3.json"
    output_file = "/mnt/petrelfs/taoye/workspace/editing/data/add/audiocaps/add_fore3/gen.txt"
    process_json_file(json_file, output_file)
