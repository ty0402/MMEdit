#!/bin/bash

data_name="add/audiocaps/add_fore3"
syn_number=10000


# can replace soundtype_dir with your own data dir,like audiocaps with segs

python jams_caps_add.py \
    --max_duration 10.0 \
    --syn_number ${syn_number} \
    --sound_type_dir  /mnt/petrelfs/taoye/workspace/data/audiotime \
    --clap_score_filter 0.3 \
    --output_jams_dir /mnt/petrelfs/taoye/workspace/editing/data/${data_name}/jams \
    --output_meta /mnt/petrelfs/taoye/workspace/editing/data/${data_name}/meta/${data_name}.json \
    --min_num_events 1 \
    --max_num_events 1 \
    --max_event_occurrence 2 \
    --max_distinct_identity 1 \
    --repeat_single_prob 0.5 \
    --times_desc_prob 0.8 \
    --min_init_onset 0.0\
    --mean_init_onset 3.0 \
    --max_init_onset 5.0 \
    --add_bg_prob 1.0 \
    --min_interval 0.2 \
    --max_interval 1.5 \
    --max_single_event_duration 5.0\
    --min_single_event_duration 2.0\
    --info_type timestamp \
    --seed 3 \
    --bg_csv_file /mnt/petrelfs/taoye/workspace/data/audiocaps/data_0.5.jsonl


python /mnt/petrelfs/taoye/workspace/editing/generate_audio_from_jams.py\
    --jams_dir /mnt/petrelfs/taoye/workspace/editing/data/${data_name}/jams/raw\
    --output_dir /mnt/petrelfs/taoye/workspace/editing/data/${data_name}/audio/raw

python /mnt/petrelfs/taoye/workspace/editing/generate_audio_from_jams.py \
    --jams_dir /mnt/petrelfs/taoye/workspace/editing/data/${data_name}/jams/edit\
    --output_dir /mnt/petrelfs/taoye/workspace/editing/data/${data_name}/audio/edit