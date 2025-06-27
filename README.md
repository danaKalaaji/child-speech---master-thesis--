# Speech Recognition Algorithm for Detecting Mispronunciation for Research with Children

An Automatic Speech Recognition (ASR) pipeline designed for improving speech assessment and phoneme-level analysis **without finetuning**. This system enhances existing ASR models for pronunciation assessment by combining audio segmentation and phonemization to evaluate speech accuracy using IPA-encoded reference texts, offering a practical solution when fine-tuning is not feasible due to limited transcription data, a common constraint in children's speech research. It provides an efficient, adaptable tool for educators and researchers.

## 🎯 Project Overview

This pipeline was developed as part of an EPFL master thesis to assist educators and clinicians in evaluating pronunciation accuracy through **binary classification** (correct/incorrect) rather than traditional transcription tasks. It enhances existing ASR models without finetuning when transcription data is unavailable, detects mispronunciations in children's speech with <5% false positives, and handles real words and pseudowords with regional accent flexibility. Features include multi-model pipeline architecture with dynamic audio segmentation, tunable threshold optimized for clinical need, REST API, batch processing, visualization tools, and real-time processing optimized for clinical use.

## 🏗️ Architecture

The pipeline enhances existing models through four main stages:

### 1. ASR Segmenter
- **Model**: `Dandan0K/Intervention-xls-FR-no-LM`
- Based on jonatasgrosman/wav2vec2-xls-r-1b-french with LM=None for hotwords integration
- Generates text transcription with start/end timestamps for each identified item

### 2. Audio Segmentation
- Dynamic audio splitting using timestamped speech segments
- Optimal segment duration: 4 seconds with 1-second padding
- Addresses training/input audio length mismatches
- 3.3% accuracy improvement for standard words, 3.6% for pseudowords

### 3. Phoneme-Level ASR
- **Model**: `Cnam-LMSSC/wav2vec2-french-phonemizer`
- Generates IPA phonetic transcriptions without language model
- Better suited for mispronunciation detection

### 4. Custom Decoding & Scoring
- Enhanced decoder for phoneme logit processing with better handling of child speech variability
- Custom error lists for subtle mispronunciation detection
- Multiple candidate consideration and flexible boundary handling
- **+15.23% accuracy improvement** over the same model without passing through this pipeline

## 📋 Installation

### Prerequisites

#### 1. NVIDIA CUDA Toolkit
Download and install the NVIDIA CUDA Toolkit:
- [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

#### 2. Create Conda Environment
```bash
# Create and activate environment
conda create --name asr python=3.10 -y
conda activate asr

# Install PyTorch with CUDA support
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c pypi cudnn -y
```

#### 3. Install Dependencies
```bash
pip install transformers
pip install flask flask-cors
pip install librosa
pip install pandas numpy matplotlib
pip install requests openpyxl
```

## 📊 Required Files

### Target Excel File (`Targets_IPA.xlsx`)
Create an Excel file with the following columns for your target words/pseudowords:

| Column | Description |
|--------|-------------|
| `Target_IPA` | Primary IPA encoding(s) of target words (space-separated) |
| `Asr_IPA_Other` | Alternative valid IPA encodings for regional accents/flexibility |
| `Errors_Letter_End` | Expected transcription errors at word endings |
| `Errors_Letter_Begining` | Expected transcription errors at word beginnings |

For more information on how to fill this file, please refer to the master thesis report p.31

## 🚀 Usage

### 1. Start the ASR Server

```bash
python asr_speech_eval.py
```

The service will start on port 5000 by default.

#### Command Line Arguments
```bash
python asr_server.py --flask_port 8070 --flask_host 0.0.0.0 \
  --target_excel_path "Targets_IPA.xlsx" \
  --target_sheet_name 0
```

**Arguments:**
- `--flask_port`: Server port (default: 8070)
- `--flask_host`: Host address (default: 0.0.0.0)
- `--id_asr_segmenter`: Segmentation model ID
- `--id_asr_phonemizer`: Phonemization model ID
- `--target_excel_path`: Path to target Excel file
- `--target_sheet_name`: Sheet name/index in Excel file

### 2. Process Audio Files

#### Single File Processing
```python
from asr_client import process_single_audio_file

result = process_single_audio_file(
    audio_file_path="path/to/audio.wav",
    reference_text="bonjour comment allez vous",
    server_url="http://localhost:8070"
)

print(f"ASR Score: {result['asr_score']}")
print(f"ASR Decode: {result['asr_decode']}")
```

#### Batch Processing
```python
import pandas as pd
from asr_client import process_dataset

# Prepare DataFrame
df = pd.DataFrame({
    'filepath': ['audio1.wav', 'audio2.wav'],
    'reference_text': ['bonjour monde', 'comment allez vous']
})

results_df = process_dataset(
    df=df,
    server_url="http://localhost:8070",
    output_file="results.csv"
)
```

### 3. API Usage

#### Health Check
```http
GET /health
```
Response: `{"status": "healthy"}`

#### ASR Pipeline
```http
POST /asr_pipeline
Content-Type: application/json
```

**Request Body:**
```json
{
    "reference_text": "bonjour comment allez vous",
    "audio": "base64_encoded_audio_data",
    "target_duration_segment": 4.0,
    "min_duration_segment": 3.0,
    "threshold_diff": 2.0,
    "max_pred_per_word": 25,
    "k": 20,
    "overlap": true,
    "segmenter": true
}
```

**Response:**
```json
{
    "asr_score": [1, 1, 0, 1],
    "asr_decode": [["bɔ̃ʒuʁko"], ["uʁkɔmɑ̃"], [], ["levu"]]
}
```

## 📈 Visualization

### Overall Results Analysis
```python
from asr_client import plot_overall_results

plot_overall_results(
    result_df=results_df,
    reference_score_col='ground_truth_score',
    asr_score_column='asr_score',
    tasks_col='task_id',
    title='ASR Performance Analysis'
)
```

### Item-level Analysis
```python
from asr_client import plot_item_level_results

plot_item_level_results(
    result_df=results_df,
    reference_text_col='reference_text',
    reference_score_col='ground_truth_score',
    tasks_col='task_id',
    only_keep_task_id='task_1',
    title='Word-level Performance'
)
```

## ⚙️ Configuration Parameters

### Segmentation Parameters
- `target_duration_segment` (float): Target segment duration (default: 4.0s)
- `min_duration_segment` (float): Minimum segment duration (default: 3.0s)
- `overlap` (bool): Allow overlapping segments (default: True)
- `segmenter` (bool): Use segmenter for longer audio (default: True)

### Decoding Parameters
- `threshold_diff` (float): Threshold for alternative predictions (default: 2.0)
- `max_pred_per_word` (int): Maximum predictions per word (default: 25)
- `k` (int): Number of alternative phonemes (-1 for full vocabulary, default: 20)


## 🔧 Output Format

### ASR Score
Binary list indicating correctness for each word:
- `1`: Word correctly pronounced
- `0`: Word incorrectly pronounced or not detected

### ASR Decode
List of phonetic predictions for each word:
- Empty list `[]`: No valid prediction found
- List of strings: Phonetic transcription variants


## Dataset & Evaluation
- **18 reading lists** with 96 items categorized by difficulty
- **~6 hours** of children's speech audio
- **Clinician-annotated ground truth** with 0-2 accuracy scale
