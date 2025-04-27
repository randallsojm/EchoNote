from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import os

def load_diarization(file_path):
    """Loads diarization data from a file in the format:
       [start_time] --> [end_time] | Speaker [speaker_number]
    """
    annotation = Annotation()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(" | ")
            if len(parts) != 2:
                continue  # Skip invalid lines
           
            time_range, speaker_label = parts
            start_time, end_time = map(float, time_range.split(" --> "))
            speaker = speaker_label.strip()


            # Add to annotation
            annotation[Segment(start_time, end_time)] = speaker


    return annotation


def evaluate_diarization(ground_truth_file, predicted_file):
    """Computes Diarization Error Rate (DER)"""
    ground_truth = load_diarization(ground_truth_file)
    predicted = load_diarization(predicted_file)


    # Initialize DER metric
    metric = DiarizationErrorRate()
    # Compute DER
    der_score = metric(ground_truth, predicted, detailed=True)
    # Additional stats if available (e.g., speaker overlaps, etc.)

    # Return all stats
    return der_score

def evaluate_all_files(original_data_folder, predicted_data_folder, output_file):
    """Evaluates all files in fdiff folder vs split_diarization folder."""
    original_data_files = sorted(os.listdir(original_data_folder))
    predicted_data_files = sorted(os.listdir(predicted_data_folder))

    if len(original_data_files) != len(predicted_data_files):
        print("Warning: Number of files in fdiff and split_diarization do not match.")
    # Open output file for writing results
    with open(output_file, 'w') as f:
        for original_data_file, predicted_data_file in zip(original_data_files, predicted_data_files):
            original_data_path = os.path.join(original_data_folder, original_data_file)
            predicted_data_path = os.path.join(predicted_data_folder, predicted_data_file)

            # Evaluate diarization for this file pair
            evaluation_score = evaluate_diarization(original_data_path, predicted_data_path)

            # Write the comparison results
            f.write(f"Evaluating {original_data_file} vs {predicted_data_file}:\n")
            f.write(f"DER Score: {evaluation_score}\n\n")
            print(f"Evaluation output file saved to {output_file}")
            

original_data_folder = "original_data"
predicted_data_folder = "predicted_data"
output_file = "evaluation_DER_results.txt"

evaluate_all_files(original_data_folder, predicted_data_folder, output_file)