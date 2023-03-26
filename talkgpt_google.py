"""
Voice-based Q&A System using Google Cloud Speech-to-Text, Google Cloud Text-to-Speech, and OpenAI GPT-3.

This script is a voice-based question and answer system that records user's voice input, transcribes it, generates a response using OpenAI GPT-3, and converts the response to speech.

Functions:
    - record_audio(): Record audio for 5 seconds using PyAudio and return the audio data in binary format.
    - speech_to_text(audio_data): Transcribe the given audio data using Google Cloud Speech-to-Text and return the transcribed text.
    - chat_gpt_response(text_input): Generate a response to the given text input using OpenAI's GPT-3 language model and return the generated response.
    - text_to_speech(text): Convert the given text to speech using Google Cloud Text-to-Speech and save it as an MP3 file (output.mp3).
    - main(): The main function that ties all the other functions together.
"""

import os
import openai
import pyaudio
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts
from playsound import playsound
from time import sleep

# Initialize the Google Cloud Speech-to-Text client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'talk-gpt-381804-296a943be00a.json' 
speech_client = speech.SpeechClient()
tts_client = tts.TextToSpeechClient()

# Initialize the OpenAI GPT-3 client
openai.api_key = 'your-own-open-ai-key'

def record_audio():
    """
    Record audio for 5 seconds using PyAudio and return the audio data in binary format.
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
        frames.append(data)

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return b''.join(frames)

def speech_to_text(audio_data):
    """
    Transcribe the given audio data using Google Cloud Speech-to-Text and return the transcribed text.
    :param audio_data: Audio data in binary format.
    :return: Transcribed text or None if the transcription was not successful.
    """
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
    )

    response = speech_client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    return None

def chat_gpt_response(text_input):
    """
    Generate a response to the given text input using OpenAI's GPT-3 language model and return the generated response.
    :param text_input: The text input to generate a response for.
    :return: Generated response from the GPT-3 language model.
    """
    prompt = f"{text_input}"
    model_engine = "text-davinci-002"
    response = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=150, n=1, stop=None, temperature=0.5)
    return response.choices[0].text.strip()

def text_to_speech(text):
    """
    Convert the given text to speech using Google Cloud Text-to-Speech and save it as an MP3 file (output.mp3).
    :param text: The text to be converted to speech.
    """
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(language_code='en-US', ssml_gender=tts.SsmlVoiceGender.FEMALE)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print(f'Generated speech for: "{text}"')

def main():
    """
    The main function that ties all the other functions together.
    """
    audio_data = record_audio()
    text_input = speech_to_text(audio_data)

    if text_input is not None:
        print(f"Recognized speech: {text_input}")
        gpt_response = chat_gpt_response(text_input)
        text_to_speech(gpt_response)
        print(f"ChatGPT response: {gpt_response}")
        # for playing note.mp3 file
        sleep(1)
        playsound('output.mp3')
        print('playing sound using  playsound')
        print("You can listen to the response in the 'output.mp3' file.")
    else:
        print("Could not recognize the speech. Please try again.")

if __name__ == "__main__":
    main()