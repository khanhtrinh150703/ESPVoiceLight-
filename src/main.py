import sounddevice as sd
from transformers import pipeline
import numpy as np
import wavio
import os
import keyboard
import paho.mqtt.client as mqtt
import re
import time
from queue import Queue
from threading import Thread

# Initialize pipeline
try:
    transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# MQTT configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "/speech/command"
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # Seconds per audio chunk for wake phrase detection
SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
SILENCE_TIMEOUT = 3  # Seconds of silence before stopping command recording

def process_audio_chunk(audio, sample_rate):
    """Save audio to a temporary file and transcribe it."""
    audio_file = "temp_audio.wav"
    wavio.write(audio_file, audio, sample_rate, sampwidth=2)
    try:
        result = transcriber(audio_file)
        text = result.get("text", "").lower().strip()
        os.remove(audio_file)
        return text
    except Exception as e:
        print(f"Error processing audio: {e}")
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return ""

def is_silent(audio, threshold=SILENCE_THRESHOLD):
    """Check if audio is silent based on RMS energy."""
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold

def listen_for_wake_phrase(sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
    """
    Continuously listen for the wake phrase 'xin chào' in real time.
    Returns True if detected, False if interrupted.
    """
    print("Listening for wake phrase 'xin chào'... Press 'q' to quit.")
    audio_buffer = Queue()
    stop_flag = [False]

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        audio_buffer.put(indata.copy())

    def process_buffer():
        """Process audio chunks from the buffer."""
        chunk_samples = int(sample_rate * chunk_duration)
        current_chunk = np.zeros((chunk_samples, 1), dtype='int16')
        current_pos = 0

        while not stop_flag[0]:
            if audio_buffer.empty():
                time.sleep(0.01)
                continue

            indata = audio_buffer.get()
            samples = indata.shape[0]

            # Fill current chunk
            for i in range(samples):
                if current_pos >= chunk_samples:
                    # Process full chunk
                    text = process_audio_chunk(current_chunk, sample_rate)
                    print(f"Detected: {text}")
                    if "xin chào" in text:
                        stop_flag[0] = True
                        break
                    # Slide window: keep last half, reset position
                    current_chunk[:chunk_samples//2] = current_chunk[chunk_samples//2:]
                    current_pos = chunk_samples // 2

                current_chunk[current_pos] = indata[i]
                current_pos += 1

    # Start audio stream
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=audio_callback)
    stream.start()

    # Start processing thread
    processor = Thread(target=process_buffer)
    processor.start()

    # Check for 'q' key to stop
    while not stop_flag[0]:
        if keyboard.is_pressed("q"):
            print("Exiting wake phrase detection...")
            stop_flag[0] = True
            stream.stop()
            stream.close()
            processor.join()
            return False
        time.sleep(0.1)

    # Clean up
    stream.stop()
    stream.close()
    processor.join()
    return True

def record_command(sample_rate=SAMPLE_RATE):
    """
    Record a command until silence is detected or timeout.
    Returns the audio file path or None if failed.
    """
    print("Recording command... (stops on silence)")
    audio_data = []
    silence_start = None
    chunk_samples = int(sample_rate * 0.5)  # 0.5-second chunks for silence detection

    def callback(indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        audio_data.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback)
    stream.start()

    while True:
        if len(audio_data) * chunk_samples >= sample_rate * 30:  # Max 30 seconds
            print("Command recording timed out.")
            break
        if keyboard.is_pressed("q"):
            print("Command recording stopped by user.")
            break

        # Check latest chunk for silence
        if audio_data:
            latest_chunk = audio_data[-1]
            if is_silent(latest_chunk):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT:
                    print("Silence detected, stopping command recording.")
                    break
            else:
                silence_start = None

        time.sleep(0.1)

    stream.stop()
    stream.close()

    if not audio_data:
        print("No command audio recorded!")
        return None

    audio = np.concatenate(audio_data, axis=0)
    audio_file = "temp_command.wav"
    wavio.write(audio_file, audio, sample_rate, sampwidth=2)
    return audio_file

def process_text(text):
    """Process transcribed text into commands."""
    text_lower = text.lower()
    if re.search(r"(bật|mở)\s*(đèn|light)", text_lower):
        return "turn on"
    elif re.search(r"(tắt|đóng)\s*(đèn|light)", text_lower):
        return "turn off"
    elif re.search(r"(mở|bật)\s*(cửa|door)", text_lower):
        return "open door"
    elif re.search(r"(đóng|tắt)\s*(cửa|door)", text_lower):
        return "close door"
    elif re.search(r"(tăng|sáng)\s*(đèn|light)", text_lower):
        return "increase light"
    elif re.search(r"(giảm|tối)\s*(đèn|light)", text_lower):
        return "decrease light"
    return text

def main():
    mqtt_client.loop_start()
    try:
        while True:
            # Listen for wake phrase
            if not listen_for_wake_phrase():
                break  # Exit if 'q' was pressed

            # Record command
            audio_file = record_command()
            if not audio_file:
                continue

            # Process command
            print("Processing command...")
            try:
                result = transcriber(audio_file)
                transcribed_text = result["text"]
                print("Transcribed text:", transcribed_text)

                command = process_text(transcribed_text)
                mqtt_client.publish(MQTT_TOPIC, command)
                print(f"Message sent to MQTT topic '{MQTT_TOPIC}': {command}")
            except Exception as e:
                print(f"Error processing command audio: {e}")

            # Clean up
            if os.path.exists(audio_file):
                os.remove(audio_file)

            print("Ready for next wake phrase or press 'q' to quit.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()