from espnet2.bin.asr_inference import Speech2Text
import argparse
import numpy as np
import wave, time, os, glob

asr_model_path="exp/asr_train_ksc2_raw_ksc2_char_sp"
lm_model_path="exp/lm_train_lm_ksc2_char"
wav_file='path_to_wav_to_be_recognized'

train_config=asr_model_path + "/config.yaml"
model_file=asr_model_path + "/valid.acc.ave.pth"

lm_config = lm_model_path + "/config.yaml"
lm_file = lm_model_path + "/valid.loss.ave.pth"

speech2text = Speech2Text(
    asr_train_config=train_config,
    asr_model_file=model_file,
    lm_train_config=lm_config,
    lm_file=lm_file,
    token_type=None,
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=10,
    ctc_weight=0.5,
    lm_weight=0.3,
    penalty=0.0,
    nbest=1,
    device = "cpu"
)

def recognize(wavfile):
    timer = time.perf_counter()
    with wave.open(wavfile, 'rb') as wavfile:
        ch=wavfile.getnchannels()
        bits=wavfile.getsampwidth()
        rate=wavfile.getframerate()
        nframes=wavfile.getnframes()
        buf = wavfile.readframes(-1)
        data=np.frombuffer(buf, dtype='int16')
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    results = speech2text(speech)
    return results[0][0]

file = '01_04_000.wav'
timer = time.perf_counter()
repeat = 10
for i in range(repeat):
    recognize(file)
print('avg time passed:', (time.perf_counter()-timer)/10)