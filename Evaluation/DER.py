import whisper
from pyannote.audio import Pipeline
from utils import diarize_text
#python pyannote_whisper_DER_offline.py
import torch
#Downloads/testing/bin/speaker_diarisation/Test_convo
def seconds_to_minutes(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes}.{seconds:02}"

def diarisation():
    # Load the CallHome dataset (ensure authentication is successful)
    dataset = "Test_convo_audio.wav"
    model = whisper.load_model("whisper/tiny.en.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.float32)
    pipeline = Pipeline.from_pretrained("config.yaml")
    asr_result = model.transcribe(dataset)
    diarization_result = pipeline(dataset)
    final_result = diarize_text(asr_result, diarization_result)


    speaker_mapping = {}
    speaker_count = 0
    # Assuming speaker_mapping and output_lines are already defined

    # Define a dictionary to replace specific speakers with names
    speaker_replacement = {'A': 'Randall', 'B': 'Helin'}
    prev_speaker = None
    prev_start = None
    prev_end = None
    merged_output = []
    diarisation_output = []

    for seg, speaker, transcript in final_result:
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = chr(65 + speaker_count)  # Map SPEAKER_00 -> A, SPEAKER_01 -> B, etc.
            speaker_count += 1
        
        speaker_label = speaker_mapping[speaker]
        
        # Replace the speaker label with the defined name if it's A or B
        if speaker_label in speaker_replacement:
            speaker_label = speaker_replacement[speaker_label]

        timestart = seconds_to_minutes(seg.start)
        timeend = seconds_to_minutes(seg.end)
        speaker = speaker_label


        if speaker == prev_speaker and prev_end == timestart:
            # Extend the previous segment's end time
            prev_end = timeend
            prev_transcript += " " + transcript  # Append with a space
        else:
            # Save the previous merged segment before starting a new one
            
            if prev_speaker is not None:
                merged_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n")
                diarisation_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n")

            # Start a new segment
            prev_speaker = speaker
            prev_start = timestart
            prev_end = timeend
            prev_transcript = transcript  # Reset transcript
        

    # Append the last segment
    if prev_speaker is not None:
        merged_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n")
        diarisation_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n")
    
    file_path = "pyannote_whisper_text_output.txt"
    with open(file_path, "w") as file:
        file.write("".join(merged_output))

    with open(r"predicted_data/formatted_pyannote_whisper_output.txt", "w") as file:
        for line in diarisation_output:
            file.write(line)


diarisation()
                           

