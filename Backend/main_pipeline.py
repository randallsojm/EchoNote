from speaker_diarisation import speaker_diarisation
from LLM_phase_2_llama3_docx import main_LLM
from TTS_phase_3_offline import main_TTS
import sys
import os

def main_pipeline(audio_file_path, meeting_starttime, meeting_endtime):
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found", file=sys.stderr)
        sys.exit(1)
    transcript_path, all_chunks_output = speaker_diarisation(audio_file_path, 180)
    row_idx = 0
    docx_path = 'processed/Minutes_Echonote.docx'
    podcast_text = main_LLM(meeting_starttime, meeting_endtime, transcript_path, row_idx, docx_path, all_chunks_output)
    main_TTS(podcast_text, "processed/Podcast.wav")

audio_file_path = sys.argv[1]
print("audio_file:", audio_file_path)
meeting_starttime = sys.argv[2]
print("meeting_starttime:", meeting_starttime)
meeting_endtime = sys.argv[3]
print("meeting_endtime:", meeting_endtime)
main_pipeline(audio_file_path, meeting_starttime, meeting_endtime)