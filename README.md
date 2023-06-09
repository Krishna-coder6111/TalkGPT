# TalkGPT: ChatGPT Voice Assistant

I came up with this idea when I was looking at TalkBack the android accessibility feature for disabled people.
This speech-based AI assistance tool can help various typed of disabled individuals, including people with mobility impairments, visual impairments, cognitive impairments, speech impairments, hearing impairments etc.

This repository contains two Python scripts that demonstrate voice-based interaction with ChatGPT using different speech-to-text and text-to-speech libraries. The main script uses Google Cloud Speech-to-Text and Text-to-Speech, while the alternative script uses Mozilla's DeepSpeech and Festival TTS.

## Dependencies
### Main Script
- openai
- google-cloud-speech
- google-cloud-texttospeech
- pyaudio
- playsound
- time
### Alternative Script
- deepspeech
- pyaudio
- numpy
- Festival TTS (external program)
## Setup
Clone this repository:
```bash
git clone https://github.com/Krishna-coder6111/TalkGPT.git
cd TalkGPT
```
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
Follow the instructions for each script to set up the respective API keys, credentials, and models:
### Main Script
- Create a Google Cloud account and set up a project.
- Enable the Google Cloud Speech-to-Text API and Text-to-Speech API for your project.
- Create a service account key and download the JSON file.
- Replace the path-to-your-file.json in the script with the path to your downloaded JSON file.
- Set the environment variable GOOGLE_APPLICATION_CREDENTIALS in the script with the path to your JSON file.
- Get your OpenAI API key and replace the 'your-own-open-ai-key' in the script with your key.
### Alternative Script
Download the DeepSpeech model and scorer files from the DeepSpeech GitHub repository.
Replace the deepspeech-0.9.3-models.pbmm and deepspeech-0.9.3-models.scorer in the script with the paths to your downloaded files.
Install the Festival TTS program on your system.
## Usage
### Main Script (Google Cloud Speech-to-Text and Text-to-Speech)
Run the main script with the following command:

```bash
python talkgpt_google.py
```
### Alternative Script (DeepSpeech and Festival TTS)
Run the alternative script with the following command:

```bash
python talkgpt_free.py
```
### For both scripts:

Speak when prompted to record your question or statement.
The script will process your speech, send it to ChatGPT, and play back the response using the respective text-to-speech library.

## License
This project is licensed under the MIT License.
