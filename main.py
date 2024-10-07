import asyncio
import os
import sys
import threading
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from fireworks.client import Fireworks
from groq import Groq
from pydub import AudioSegment
import soundfile as sf
import io

load_dotenv()


class VoiceChat:
    def __init__(self):
        self.fireworks = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))
        self.eleven_labs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Audio recording settings
        self.sample_rate = 44100  # Standard CD quality
        self.recording_duration = 10  # seconds
        self.recording = False
        self.audio_data = None

        # VAD settings for future use
        self.vad = webrtcvad.Vad(3)
        self.frame_duration = 30  # ms
        self.buffer = deque(maxlen=int(self.sample_rate * 3))
        self.is_speaking = False
        self.silence_threshold = 30
        self.silence_counter = 0

        # State management for loop
        self.should_stop = False
        self.current_audio_level = 0

        # Voice settings for ElevenLabs
        self.voice_settings = VoiceSettings(
            stability=0.1,
            similarity_boost=0.3,
            style=0.2,
        )

    def print_status(self, message, end="\n"):
        """Print status messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", end=end)
        sys.stdout.flush()

    def update_audio_level_indicator(self, level):
        """Display a visual audio level indicator"""
        bars = int(level * 50)  # Scale progress bar to 50 characters
        indicator = "█" * bars + "▒" * (50 - bars)
        self.print_status(f"\rAudio Level: |{indicator}| {level:.2f}", end="")

    def frame_generator(self, audio_data, frame_duration_ms):
        n = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        while offset + n <= len(audio_data):
            yield audio_data[offset : offset + n]
            offset += n

    def record_audio(self):
        self.print_status("Recording...")
        self.audio_data = sd.rec(
            int(self.recording_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
        )
        sd.wait()
        self.print_status("Recording finished.")
        self.recording = False

        sf.write("user_audio.wav", self.audio_data, self.sample_rate)
        self.print_status("Audio file saved as user_audio.wav")

    async def transcribe_audio(self, audio_file_path):
        try:
            self.print_status("Sending audio to Groq...")
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    response_format="json",
                    language="en",
                    temperature=0.0,
                )

            if hasattr(transcription, "text"):
                return transcription.text
            elif isinstance(transcription, dict) and "text" in transcription:
                return transcription["text"]
            else:
                self.print_status(f"Unexpected transcription format: {transcription}")
                return str(transcription)

        except Exception as e:
            self.print_status(f"Error in transcription: {e}")
            raise

    async def get_llm_response(self, text):
        self.print_status("Generating response...")
        full_response = ""
        print("\nAssistant: ", end="")

        try:
            response = self.fireworks.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-3b-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and engaging conversational AI assistant. Respond naturally and concisely.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=1000,
                temperature=0.7,
                stream=True,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)

            print()  # print /n after response
            return full_response

        except Exception as e:
            self.print_status(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response."

    async def speak_response(self, text):
        self.print_status("Converting text to speech...")
        try:
            audio_stream = self.eleven_labs.text_to_speech.convert_as_stream(
                model_id=os.getenv("ELEVENLABS_MODEL_ID"),
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "Josh"),
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text,
                voice_settings=self.voice_settings,
            )

            self.print_status("Playing response...")

            audio_data = b"".join(chunk for chunk in audio_stream if chunk)

            # Convert MP3 to WAV
            with io.BytesIO(audio_data) as mp3_file:
                with io.BytesIO() as wav_file:
                    audio = AudioSegment.from_mp3(mp3_file)
                    audio.export(wav_file, format="wav")
                    wav_file.seek(0)

                    # Read the WAV file
                    with sf.SoundFile(wav_file) as sound_file:
                        data = sound_file.read()
                        samplerate = sound_file.samplerate

            # Play the audio
            sd.play(data, samplerate)
            sd.wait()

        except Exception as e:
            self.print_status(f"Error in text-to-speech: {e}")

    def start_recording(self):
        self.recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_recording(self):
        if self.recording:
            sd.stop()
        if hasattr(self, "audio_thread"):
            self.audio_thread.join(timeout=1.0)
        return self.audio_data

    async def chat_loop(self):
        self.print_status("Voice Chat System Started")
        self.print_status("Press Ctrl+C to exit at any time")

        while not self.should_stop:
            try:
                self.print_status("\n" + "=" * 50)
                self.print_status("Ready to listen! Press Enter to start recording...")
                input()

                self.start_recording()
                self.print_status(f"Recording for {self.recording_duration} seconds...")

                # Wait for recording to finish
                while self.recording and not self.should_stop:
                    await asyncio.sleep(0.1)

                audio_data = self.stop_recording()

                if audio_data is None or len(audio_data) == 0:
                    self.print_status("No audio recorded. Please try again.")
                    continue

                transcription = await self.transcribe_audio("user_audio.wav")
                self.print_status(f"\nYou said: {transcription}")

                if "goodbye" in transcription.lower():
                    self.print_status("Goodbye detected, ending conversation...")
                    break

                response = await self.get_llm_response(transcription)
                await self.speak_response(response)

            except KeyboardInterrupt:
                self.print_status("\nStopping voice chat system...")
                self.should_stop = True
                break
            except Exception as e:
                self.print_status(f"\nError: {e}")
                continue


if __name__ == "__main__":

    async def main():
        chat_system = VoiceChat()
        await chat_system.chat_loop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting voice chat system...")
