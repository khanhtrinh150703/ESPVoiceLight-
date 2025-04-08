import sounddevice as sd
from transformers import pipeline
import numpy as np
import wavio
import os
import keyboard
import paho.mqtt.client as mqtt
import re

# Khởi tạo pipeline
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")

# Cấu hình MQTT
MQTT_BROKER = "localhost"  # Thay bằng IP của broker nếu không chạy local
MQTT_PORT = 1883
MQTT_TOPIC = "/speech/command"  # Topic để gửi lệnh

# Khởi tạo client MQTT
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

def record_audio(sample_rate=16000):
    print("Press Enter to start recording...")
    keyboard.wait("enter")
    print("Recording... Press Enter again to stop.")
    
    audio_data = []
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())
    
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback)
    stream.start()
    
    keyboard.wait("enter")
    stream.stop()
    stream.close()
    print("Recording stopped.")
    
    audio = np.concatenate(audio_data, axis=0)
    audio_file = "recorded_audio.wav"
    wavio.write(audio_file, audio, sample_rate, sampwidth=2)
    return audio_file

def process_text(text):
    """Xử lý văn bản với regex và trả về lệnh phù hợp"""
    text_lower = text.lower()
    
    # Các pattern và lệnh tương ứng
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
    else:
        return text  # Nếu không khớp, gửi nguyên văn bản

def tts_press_to_talk():
    mqtt_client.loop_start()
    
    while True:
        audio_file = record_audio()
        
        print("Processing audio...")
        result = transcriber(audio_file)
        transcribed_text = result["text"]
        print("Transcribed text:", transcribed_text)
        
        # Xử lý văn bản với regex
        command = process_text(transcribed_text)
        
        # Gửi lệnh qua MQTT
        mqtt_client.publish(MQTT_TOPIC, command)
        print(f"Message sent to MQTT topic '{MQTT_TOPIC}': {command}")
        
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        print("Press 'q' to quit or Enter to record again.")
        if keyboard.is_pressed("q"):
            print("Exiting...")
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            break

if __name__ == "__main__":
    try:
        tts_press_to_talk()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()