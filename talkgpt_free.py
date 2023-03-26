"""
This script demonstrates a simple voice-based interaction with ChatGPT using DeepSpeech for speech-to-text and Festival for text-to-speech. The script records the user's voice, converts the speech to text using DeepSpeech, sends the text to ChatGPT, and plays back ChatGPT's response using Festival.

Dependencies:

deepspeech
pyaudio
numpy
Festival TTS (external program)
Usage:

Run the script with python script_name.py
Speak when prompted to record your question or statement
The script will process your speech, send it to ChatGPT, and play back the response using Festival TTS
"""

import os
import sys
import subprocess
import pyaudio
import numpy as np
import deepspeech

# Initialize the DeepSpeech model
model_file_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_file_path)

# Initialize the scorer for the model
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
model.enableExternalScorer(scorer_file_path)

def record_audio():
    """
    Record audio using PyAudio, storing it in an appropriate format for DeepSpeech processing.
    :return: numpy array containing the recorded audio data
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.concatenate(frames, axis=0)

def speech_to_text(audio_data):
    """
    Convert audio data to text using DeepSpeech.
    :param audio_data: numpy array containing audio data
    :return: string containing the recognized text
    """
    text = model.stt(audio_data)
    return text.strip()

def text_to_speech(text):
    """
    Convert text to speech using Festival TTS.
    :param text: string containing the text to convert to speech
    """
    with open('temp_text.txt', 'w') as f:
        f.write(text)

    subprocess.call(['text2wave', 'temp_text.txt', '-o', 'output.wav'])
    os.remove('temp_text.txt')
    print(f'Generated speech for: "{text}"')

def main():
    """
    Main function that records audio, processes it with DeepSpeech, sends it to ChatGPT, and plays back the response using Festival TTS.
    """
    audio_data = record_audio()
    text_input = speech_to_text(audio_data)

    if text_input:
        print(f"Recognized speech: {text_input}")
        gpt_response = chat_gpt_response(text_input)  # Assuming you still have the chat_gpt_response() function from the previous script
        text_to_speech(gpt_response)
        print(f"ChatGPT response: {gpt_response}")
        print("You can listen to the response in the 'output.wav' file.")
    else:
        print("Could not recognize the speech. Please try again.")

if __name__ == "__main__":
    main()