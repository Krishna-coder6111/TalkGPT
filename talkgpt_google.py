import os
import openai
import pyaudio
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts

# Initialize the Google Cloud Speech-to-Text client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'talk-gpt-381804-296a943be00a.json' #make google cloud acc 
speech_client = speech.SpeechClient()
tts_client = tts.TextToSpeechClient()

# Initialize the OpenAI GPT-3 client
openai.api_key = 'your_openai_api_key'

def record_audio():
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
    prompt = f"{text_input}"
    model_engine = "text-davinci-002"
    response = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=150, n=1, stop=None, temperature=0.5)
    return response.choices[0].text.strip()

def text_to_speech(text):
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(language_code='en-US', ssml_gender=tts.SsmlVoiceGender.FEMALE)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print(f'Generated speech for: "{text}"')

def main():
    audio_data = record_audio()
    text_input = speech_to_text(audio_data)

    if text_input is not None:
        print(f"Recognized speech: {text_input}")
        gpt_response = chat_gpt_response(text_input)
        text_to_speech(gpt_response)
        print(f"ChatGPT response: {gpt_response}")
        print("You can listen to the response in the 'output.mp3' file.")
    else:
        print("Could not recognize the speech. Please try again.")

if __name__ == "__main__":
    main()