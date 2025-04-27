import whisper
#python whisper_pyannote.py
import huggingface_hub
import torch
import jiwer
#Downloads/testing/bin/speaker_diarisation/Test_convo
#python pyannote_whisper_WER.py

def diarisation(HF_TOKEN):
    huggingface_hub.login(HF_TOKEN)
    # Load the CallHome dataset (ensure authentication is successful)
    model = whisper.load_model("tiny.en")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.float32)
    
    audio_path = 'Test_convo_audio.wav'

    # Save as a .wav file if needed  


    filename = "Transcription.txt"

    with open(filename, "r") as doc:
        transcript = doc.read()

    print("ground truth:", transcript)
    predicted_text = model.transcribe(audio_path)["text"]
    print("predicted text:", predicted_text)

    
    remove_punctuation = True 
    lowercase = True
    input_ref=transcript
    input_hyp=predicted_text


    if remove_punctuation == True:
        input_ref = jiwer.RemovePunctuation()(input_ref)
        input_hyp = jiwer.RemovePunctuation()(input_hyp)
    
    if lowercase == True:
        input_ref = input_ref.lower()
        input_hyp = input_hyp.lower()

    output, compares = wer(input_ref, input_hyp ,debug=True)
    file_name = "Evaluation_pyannote_whisper_WER.txt"
    with open(file_name, "w") as file:
        file.write(f"Ground Truth: {input_ref} \n\nPredicted Text: {input_hyp}")
        file.write(f"CORRECT   : {output['Cor']}\n")
        file.write(f"DELETE    : {output['Del']}\n")
        file.write(f"N SUBSTITUTE: {output['Sub']}\n")
        file.write(f"N INSERT    : {output['Ins']}\n")
        file.write(f"WER: {output['WER']}\n")
    
    print(f"WER data saved to {file_name}")



def wer(ref, hyp ,debug=False):
    r = ref.split()
    h = hyp.split()
    print("h:", h)
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        lines = []
        compares = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
                compares.append(h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
                compares.append(h[j] +  r[i])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
                compares.append(h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
                compares.append(r[i])
    if debug:
        # print("OP\tREF\tHYP")
        # lines = reversed(lines)
        # for line in lines:
        #     print(line)

        compares = reversed(compares)
        for line in compares:
          print(line, end=" ")
        # print("Ncor " + str(numCor))
        # print("Nsub " + str(numSub))
        # print("Ndel " + str(numDel))
        # print("Nins " + str(numIns))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}, compares


#@title Calculate and visualize WER { run: "auto" }
HF_TOKEN = 'YOUR API KEY'
diarisation(HF_TOKEN)

