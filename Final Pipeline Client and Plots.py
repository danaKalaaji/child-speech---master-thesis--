import requests
import base64
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################ ASR Pipeline Client #################################################

def process_single_audio_file(audio_file_path: str, 
                              reference_text: str, 
                              server_url: str = "http://localhost:8070", 
                              pipeline_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a single audio file through the ASR server. Timeout is set to 2 minutes.
    
    Args:
        audio_file_path (str): Path to the audio file
        reference_text (str): Reference text for the audio
        server_url (str, optional): URL of the ASR server, defaults to "http://localhost:8070"
        pipeline_params (dict, optional): Additional parameters for the pipeline. If not provided, defaults will be used.
        
    Returns:
        dict: Server response containing 'asr_score', 'asr_decode'
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        requests.RequestException: If server request fails
        ValueError: If server returns an error
    """
    # Validate inputs
    if not isinstance(audio_file_path, str):
        raise ValueError("Audio file path must be a string")
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    if not reference_text.strip():
        raise ValueError("Reference text cannot be empty")
    
    # Default pipeline parameters
    default_params = {
        'target_duration_segment': 4.0,
        'min_duration_segment': 3.0,
        'threshold_diff': 2.0,
        'max_pred_per_word': 25,
        'k': 20,
        'overlap': True,
        'segmenter': True
    }
    if pipeline_params:
        default_params.update(pipeline_params)
    
    try:
        # Read and encode audio file
        logger.info(f"Processing audio file: {audio_file_path}") #comment this line for shorter logs
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('ASCII')
        
        # Prepare request payload
        payload = {
            'reference_text': reference_text,
            'audio': audio_base64,
            **default_params
        }
        
        # Make request to server
        endpoint_url = f"{server_url.rstrip('/')}/asr_pipeline"
        logger.info(f"Sending request to: {endpoint_url}") #comment this line for shorter logs
        
        response = requests.post(
            endpoint_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=120  # 2 minutes timeout
        )
        
        # Check response status
        if response.status_code != 200:
            error_msg = f"Server error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise requests.RequestException(error_msg)
        
        # Parse response
        result = response.json()
        
        # Validate response structure
        if 'error' in result:
            raise ValueError(f"Server returned error: {result['error']}")
        
        if 'asr_score' not in result or 'asr_decode' not in result:
            raise ValueError("Invalid response format from server")
        
        logger.info(f"Successfully processed audio file: {audio_file_path}") #comment this line for shorter logs
        return result
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout for file: {audio_file_path}") #comment this line for shorter logs
        raise requests.RequestException("Server request timed out")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to server: {server_url}") #comment this line for shorter logs
        raise requests.RequestException(f"Cannot connect to server: {server_url}")
    except Exception as e:
        logger.error(f"Error processing {audio_file_path}: {str(e)}") #comment this line for shorter logs
        raise


def process_dataset(df: pd.DataFrame, 
                    server_url: str = "http://localhost:8070",
                    output_file: Optional[str] = "asr_results.csv", 
                    pipeline_params: Optional[Dict[str, Any]] = None,
                    asr_score_col='asr_score',
                    asr_decode_col ='asr_decode',
                    ) -> pd.DataFrame:
    """
    Process a dataset of audio files through the ASR server
    
    Args:
        df (pd.DataFrame): DataFrame with 'filepath' and 'reference_text' columns
        server_url (str, optional): URL of the ASR server, defaults to "http://localhost:8070"
        output_file (str, optional): Path to save results CSV file
        pipeline_params (dict, optional): Additional parameters for the pipeline
        
    Returns:
        pd.DataFrame: DataFrame with results including 'asr_score' and 'asr_decode'        
    Raises:
        ValueError: If required columns are missing from dataframe
        Exception: If processing fails
    """
    # Validate dataframe structure
    required_columns = ['filepath', 'reference_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataframe: {missing_columns}")
    
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    logger.info(f"Starting dataset processing with {len(df)} files")
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    
    # Create new columns for results if they don't exist
    if asr_score_col not in result_df.columns:
        result_df[asr_score_col] = None
    if asr_decode_col not in result_df.columns:
        result_df[asr_decode_col] = None

    # Process each audio file
    for index, row in result_df.iterrows():
        audio_file_path = row['filepath']
        reference_text = row['reference_text']
        
        try:
            # Process the audio file
            result = process_single_audio_file(
                audio_file_path,
                reference_text,
                server_url=server_url,
                pipeline_params=pipeline_params
            )
            
            # Update results in the dataframe
            result_df.at[index, asr_score_col] = result.get('asr_score')
            result_df.at[index, asr_decode_col] = result.get('asr_decode')
        
        except Exception as e:
            logger.error(f"Error processing file {audio_file_path}: {str(e)}")
            result_df.at[index, asr_score_col] = -1
            result_df.at[index, asr_decode_col] = -1

    # Save results to CSV if output file is specified
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_file}")
    return result_df

        
############################################ Plotting Functions #################################################

def plot_overall_results(result_df, 
                         reference_score_col, asr_score_column='asr_score', 
                         tasks_col = None, title='ASR Overall Results'):
    """
    Plots TP, FP, FN, and TN rates from ASR and reference scores.

    Parameters:
    - result_df: DataFrame containing ASR and reference scores
    - title: Plot title
    - reference_score_col: column name of the reference accuracy
    - asr_acc_column: column name of the ASR model prediction (default = 'asr_score')
    - tasks_col: column name containing the tasks IDs (optional, for per-task analysis)
    """

    # Validate input DataFrame
    if not isinstance(result_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if reference_score_col not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{reference_score_col}' column")
    if asr_score_column not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{asr_score_column}' column")
    if tasks_col is not None and tasks_col not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{tasks_col}' column for per-task analysis mode")
    
    # Filter out rows with invalid ASR score and replace 'NA' with 0
    result_df = result_df[result_df[asr_score_column] != -1].copy()
    result_df[reference_score_col] = result_df[reference_score_col].apply(
        lambda x: ' '.join(['0' if token == 'NA' else token for token in x.split()])
    )

    # Initialize dictionaries to hold counts for each config_id
    tp_counts, tn_counts, fp_counts, fn_counts, word_counts = {}, {}, {}, {}, {}


    for idx, row in result_df.iterrows():
        
        try: # Parse ASR and reference scores
            reference_score = [int(i) for i in row[reference_score_col].split() if i.isdigit()]
            # Change mapping of reference scores to match ASR scores (treat 2:'correct' and 1:'acceptable' as 1:correct)
            reference_score = [1 if val == 2 else val for val in reference_score]
            asr_score = [int(i) for i in row[asr_score_column].split() if i.isdigit()]
        except Exception as e:
            raise ValueError(f"Error parsing scores for row {idx}: {e}")
        if len(asr_score) != len(reference_score):
            raise ValueError(f"ASR and reference scores length mismatch for row {idx}: {len(asr_score)} vs {len(reference_score)}")
        
        # Initialize counts for this config_id if not already done
        config_id = row[tasks_col] if tasks_col else 'overall'
        if config_id not in tp_counts:
            length = len(asr_score)
            tp_counts[config_id] = [0] * length
            tn_counts[config_id] = [0] * length
            fp_counts[config_id] = [0] * length
            fn_counts[config_id] = [0] * length
            word_counts[config_id] = [0] * length

        # Count TP, TN, FP, FN for each word
        for i in range(len(asr_score)):
            pred = asr_score[i]
            ref = reference_score[i]

            if pred == 1 and ref == 1: # True Positive
                tp_counts[config_id][i] += 1
            elif pred == 1 and ref == 0: # False Positive
                fp_counts[config_id][i] += 1
            elif pred == 0 and ref == 1: # False Negative
                fn_counts[config_id][i] += 1
            elif pred == 0 and ref == 0: # True Negative
                tn_counts[config_id][i] += 1
            word_counts[config_id][i] += 1
    
    # Compute average rates for each config_id
    tp_rates, tn_rates, fp_rates, fn_rates = [], [], [], []
    config_ids = sorted(tp_counts.keys())
    for config_id in config_ids:
        total = sum(word_counts[config_id])
        if total == 0:
            tp_rates.append(0)
            tn_rates.append(0)
            fp_rates.append(0)
            fn_rates.append(0)
        else:
            tp_rates.append(sum(tp_counts[config_id]) / total)
            tn_rates.append(sum(tn_counts[config_id]) / total)
            fp_rates.append(sum(fp_counts[config_id]) / total)
            fn_rates.append(sum(fn_counts[config_id]) / total)

    # Plotting the results
    indices = np.arange(len(config_ids))
    plt.figure(figsize=(12.5, 6))

    plt.bar(indices, tp_rates, label='TP Rate', color='green', bottom=np.add(tn_rates, np.add(fp_rates, fn_rates)))
    plt.bar(indices, tn_rates, label='TN Rate', color='blue', bottom=np.add(fp_rates, fn_rates))
    plt.bar(indices, fp_rates, label='FP Rate', color='red', bottom=fn_rates)
    plt.bar(indices, fn_rates, label='FN Rate', color='orange')

    plt.xticks(indices, config_ids, rotation=45)
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n{title}\n{'='*len(title)}")
    for config_id, tp, tn, fp, fn in zip(config_ids, tp_rates, tn_rates, fp_rates, fn_rates):
        print(f"{config_id} → TP: {tp:.2f}, TN: {tn:.2f}, FP: {fp:.2f}, FN: {fn:.2f}")


def plot_item_level_results(result_df, reference_text_col: str, 
                            reference_score_col: str, tasks_col: str, only_keep_task_id: str,
                            asr_score_column='asr_score', title = 'ASR Item Level Results'):
    # Validate input DataFrame
    if not isinstance(result_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if reference_text_col not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{reference_text_col}' column")
    if reference_score_col not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{reference_score_col}' column")
    if asr_score_column not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{asr_score_column}' column")
    if tasks_col not in result_df.columns:
        raise ValueError(f"DataFrame must contain '{tasks_col}' column to remove unwanted tasks ids")
    
    # Filter out rows not in only_keep_tasks_id if tasks_col is provides
    result_df = result_df[result_df[asr_score_column] != -1].copy()
    try:
        # Filter the DataFrame to keep only specified task IDs from the tasks_col
        result_df = result_df[result_df[tasks_col] == only_keep_task_id]
    except Exception as e:
        raise ValueError(f"Error filtering tasks: {e}")
    
    # Filter out rows with invalid ASR score and replace 'NA' with 0
    result_df[reference_score_col] = result_df[reference_score_col].apply(
        lambda x: ' '.join(['0' if token == 'NA' else token for token in x.split()])
    )
    
    # Initialize lists to store TP, TN, FP, FN counts for each word
    all_tp, all_tn, all_fp, all_fn, word_counts = [], [], [], [], []

    for idx, row in result_df.iterrows():
        # Parse ASR and reference scores
        try:
            reference_score = [int(i) for i in row[reference_score_col].split() if i.isdigit()]
            # Change mapping of reference scores to match ASR scores (treat 2:'correct' and 1:'acceptable' as 1:correct)
            reference_score = [1 if val == 2 else val for val in reference_score]
            asr_score = [int(i) for i in row[asr_score_column].split() if i.isdigit()]
        except Exception as e:
            raise ValueError(f"Error parsing scores for row {idx}: {e}")
        if len(asr_score) != len(reference_score):
            raise ValueError(f"ASR and reference scores length mismatch for row {idx}: {len(asr_score)} vs {len(reference_score)}")

        # Initialize counts for this row
        length= len(asr_score)
        tp = [0] * len(length)
        tn = [0] * len(length)
        fp = [0] * len(length)
        fn = [0] * len(length)
        
        for i in range(len(asr_score)):
            if asr_score[i] == 1 and reference_score[i] == 1:
                tp[i] = 1
            elif asr_score[i] == 1 and reference_score[i] == 0:
                fp[i] = 1
            elif asr_score[i] == 0 and reference_score[i] == 1:
                fn[i] = 1
            elif asr_score[i] == 0 and reference_score[i] == 0:
                tn[i] = 1
        
        all_tp.append(tp)
        all_tn.append(tn)
        all_fp.append(fp)
        all_fn.append(fn)
        word_counts.append([1] * len(asr_score))  # Count each word occurrence

    # Sum the counts for each word across all rows
    tp_sums = [sum(x) for x in zip(*all_tp)]
    tn_sums = [sum(x) for x in zip(*all_tn)]
    fp_sums = [sum(x) for x in zip(*all_fp)]
    fn_sums = [sum(x) for x in zip(*all_fn)]
    word_total_counts = [sum(x) for x in zip(*word_counts)]

    # Calculate rates
    tp_rates = [tp / total for tp, total in zip(tp_sums, word_total_counts)]
    tn_rates = [tn / total for tn, total in zip(tn_sums, word_total_counts)]
    fp_rates = [fp / total for fp, total in zip(fp_sums, word_total_counts)]
    fn_rates = [fn / total for fn, total in zip(fn_sums, word_total_counts)]

    # Plotting the results
    words = np.arange(len(tp_rates))  # Word indices

    # Create a bar plot for TP, TN, FP, FN rates
    plt.figure(figsize=(13, 2.5))
    plt.bar(words, tp_rates, label='TP Rate', color='green', bottom=np.add(tn_rates, np.add(fp_rates, fn_rates)))
    plt.bar(words, tn_rates, label='TN Rate', color='blue', bottom=np.add(fp_rates, fn_rates))
    plt.bar(words, fp_rates, label='FP Rate', color='red', bottom=fn_rates)
    plt.bar(words, fn_rates, label='FN Rate', color='orange')

    plt.xlabel('')
    # add the words to the x-axis from row['reference_text']
    plt.xticks(words, [word for word in row['reference_text'].split(" ")])  
    plt.ylabel('Rate')
    plt.title(title)
    plt.title('TP, TN, FP, FN Rates per Word')
    # put legend on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='black')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axhline(y=0.5, color='black', linestyle='-', linewidth=1)
    plt.show()

    # print tp, tn, fp, fn rates for each words (rounded to 1 decimals)
    for word, tp, tn, fp, fn in zip([word for word in row['reference_text'].split(" ")], tp_rates, tn_rates, fp_rates, fn_rates):
        print(f'{word}: TP = {tp:.2f}, TN = {tn:.2f}, FP = {fp:.2f}, FN = {fn:.2f}')




    