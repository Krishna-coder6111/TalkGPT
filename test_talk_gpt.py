"""
Test class for talk gpt using google apis.
"""

import unittest
import os
from talkgpt_google import record_audio, speech_to_text, chat_gpt_response, text_to_speech

class TestChatGPT(unittest.TestCase):

    def test_record_audio(self):
        audio_data = record_audio()
        self.assertIsNotNone(audio_data, "Audio recording should return data")

    def test_speech_to_text(self):
        # short audio file in the LINEAR16 format - 16000 range, mono channel
        test_audio_file = 'india_capital16000mono.wav'
        with open(test_audio_file, 'rb') as f:
            test_audio_data = f.read()
        text = speech_to_text(test_audio_data)
        self.assertIsNotNone(text, "Speech-to-Text conversion should return text")

    def test_chat_gpt_response(self):
        test_input = "What is the capital of France?"
        response = chat_gpt_response(test_input)
        self.assertIsNotNone(response, "GPT-3 should return a response")

    def test_text_to_speech(self):
        test_text = "This is a test."
        text_to_speech(test_text)
        self.assertTrue(os.path.exists('output.mp3'), "Text-to-Speech should generate an output file")

if __name__ == '__main__':
    unittest.main()