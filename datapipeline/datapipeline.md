

# 🎧 MMEdit Data Pipeline

## Overview
This repository implements an automated data generation pipeline for audio editing tasks. We support **6 core editing operations**. 

The pipeline utilizes the **Scaper** library to programmatically mix acoustic scenes from **AudioCaps** (background) with sound events from **AudioTime** (foreground), generating high-quality instruction-audio pairs.


## Data Preparation

Ensure the source datasets are correctly prepared and placed in the expected locations. This pipeline requires:

- **Background (scene bed):** **AudioCaps* 

- **Foreground (events to add):** \
 **AudioTime** https://zeyuxie29.github.io/AudioTime/  \
**AudioCaps segs** (pre-cut AudioCaps clips with segment annotations).  
  Reference: https://github.com/wsntxxn/TextToAudioGrounding


---

## 🚀 "Addition"


### 1. Audio Synthesis (Scaper)

We use a shell script to orchestrate the `scaper` logic. This process generates **JAMS** annotation files (containing timing, SNR, and source info) and renders the final **synthesized audio**.

Run the generation script:

```bash
cd scaper_scripts/add/

# Run the synthesis pipeline
# This example executes the logic to mix AudioCaps and AudioTime
bash add.sh

```

### 2. Instruction Generation

Once audio synthesis is complete, we convert the structured JAMS metadata into natural language editing instructions (e.g., *"Add a dog barking sound..."*).

```bash
# Example command for caption generation
python add_fore.py 

```

## "Removal"
will coming soon
---

## 📂 Directory Structure

```text
MMEdit/datapipeline/
├── datasets/
│   ├── AudioCaps/       # Background source
│   └── AudioTime/       # Foreground source
├── scaper_scripts/
│   ├── add/             # "Add" operation workspace
│   │   ├── add.sh       # Entry script
│   │   └── jams_caps_add.py
    ├── remove/          # Other operations...  
        └── ...


```


```