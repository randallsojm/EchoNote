import whisper
from pyannote.audio import Pipeline
from utils import diarize_text
import torch

def seconds_to_minutes(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes}.{seconds:02}"

def chunk_by_time_intervals(segments, interval_seconds):
    """Chunks transcript segments into intervals based on time."""
    chunks = []
    current_chunk = []
    chunk_start_time = None

    for seg, speaker, transcript in segments:
        start_time = seg.start

        # If the chunk is empty, set its start time
        if chunk_start_time is None:
            chunk_start_time = start_time

        # If the segment exceeds the time interval, start a new chunk
        if start_time - chunk_start_time >= interval_seconds:
            chunks.append(current_chunk)
            print("chunk:", chunks)
            current_chunk = []
            chunk_start_time = start_time  # Reset chunk start time

        current_chunk.append((seg, speaker, transcript))

    # Append the last chunk if there's remaining data
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
def speaker_diarisation(audio_file_path, chunk_interval_seconds):
    # Load the CallHome dataset (ensure authentication is successful)
    dataset = audio_file_path
    # print("audio_file: no audio file")
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
    # for speaker_mapping_dict in speaker_mapping_list:
    #     speaker_name = speaker_mapping_dict.get('name', '')
    #     speaker_label = speaker_mapping_dict.get('label', '')
    #     speaker_replacement[speaker_label] = speaker_name
    prev_speaker = None
    prev_start = None
    prev_end = None
    prev_transcript = ""
    merged_output = []
    diarisation_output = []
    segments_for_chunking = []
    

    for seg, speaker, transcript in final_result:
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = chr(65 + speaker_count)  # Map SPEAKER_00 -> A, SPEAKER_01 -> B, etc.
            speaker_count += 1
        
        speaker_label = speaker_mapping[speaker]
        speaker = speaker_label

        segments_for_chunking.append((seg, speaker, transcript))
    print("segments_for_chunking:", segments_for_chunking)

    # Perform chunking based on the specified interval
    chunked_segments = chunk_by_time_intervals(segments_for_chunking, chunk_interval_seconds)


    all_chunks_output = []
    for i, chunk in enumerate(chunked_segments):
        print(f"Diarising chunk {i}")
        chunk_output = []
        prev_speaker = None
        prev_start = None
        prev_end = None
        prev_transcript = ""
        
        for seg, speaker, transcript in chunk:
            timestart = seconds_to_minutes(seg.start)
            timeend = seconds_to_minutes(seg.end)

            if speaker == prev_speaker and prev_end == timestart:
                # Extend the previous segment's end time
                prev_end = timeend
                prev_transcript += " " + transcript  # Append with a space
            else:
                # Save the previous merged segment before starting a new one
                if prev_speaker is not None:
                    merged_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n")
                    diarisation_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n")
                    chunk_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n")
                prev_speaker = speaker
                prev_start = timestart
                prev_end = timeend
                prev_transcript = transcript  # Reset transcript

        # Append the last segment in the chunk
        if prev_speaker is not None:
            merged_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n")
            diarisation_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n")
            chunk_output.append(f"{prev_start} --> {prev_end} | {prev_speaker}\n{prev_transcript}\n\n")
        all_chunks_output.append(chunk_output)
        print(f"chunk_output {i}: {chunk_output}")
        print("all_chunks_output:", all_chunks_output)


    file_path = "Transcript.txt"
    with open(file_path, "w") as file:
        file.write("".join(merged_output))

    with open(r"predicted_data/formatted_pyannote_whisper_output.txt", "w") as file:
        for line in diarisation_output:
            file.write(line)
    print("speaker diarisation script run completed")
    return file_path, all_chunks_output

# speaker_diarisation("Test_convo_audio.wav", 180)
# Get the file path passed from Node.js
# audio_file_path = sys.argv[1]
# print("audio_file:", audio_file_path)
# if not os.path.exists(audio_file_path):
#     print(f"Error: Audio file '{audio_file_path}' not found", file=sys.stderr)
#     sys.exit(1)
# result = speaker_diarisation(audio_file_path)

    

                           
