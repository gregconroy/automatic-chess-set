import pyaudio
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import time
import io
import os

class AudioProcessor:
    def __init__(self, channels=1, rate=44100, format=pyaudio.paInt16, chunk=64,
                 threshold=4, silence_duration=1, max_amplitude=2**15, min_clip_duration=2, start_padding=100):
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.channels = channels
        self.rate = rate
        self.format = format
        self.chunk = chunk
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.max_amplitude = max_amplitude
        self.min_clip_duration = min_clip_duration
        self.start_padding = start_padding

        self.audio_data = io.BytesIO()
        self.voice_active = False
        self.clip_start = 0
        self.clip_end = 0
        self.silent_start_time = None
        self.audio_start = 0

        # Create the output directory if it doesn't exist
        self.output_dir = "./Clips/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.stream = None
        self.detected_clip = False  # Flag to indicate if a clip has been detected

    def start_listening(self):
        # Open audio stream
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  stream_callback=self.callback)

        # Start the stream
        self.stream.start_stream()
        self.audio_start = time.time()
        print("Listening...")

    def stop_listening(self):
        if self.stream:
            # Stop the stream
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print("Stopped listening.")

    def callback(self, in_data, frame_count, time_info, flag):
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        amplitude = 100 * np.abs(audio_chunk).mean() / self.max_amplitude

        if amplitude > self.threshold:
            if not self.voice_active:
                print("active")
                self.voice_active = True
                self.silent_start_time = None
                self.clip_start = time.time()
            else:
                self.silent_start_time = time.time()

        else:
            if self.voice_active:
                if self.silent_start_time is None:
                    self.silent_start_time = time.time()

                if time.time() - self.silent_start_time > self.silence_duration:
                    print("inactive")
                    self.clip_end = time.time()
                    clip_duration = self.clip_end - self.clip_start

                    if clip_duration > self.min_clip_duration:
                        print(f'Clip duration: {clip_duration}')
                        print('Spoken move detected')
                        self.save_detected_clip(self.clip_start, self.clip_end)

                    self.voice_active = False
                    self.clip_start = 0
                    self.clip_end = 0
                    self.silent_start_time = None
                    self.audio_data = io.BytesIO()
                    self.audio_start = time.time()

        self.audio_data.write(in_data)
        return in_data, pyaudio.paContinue

    def save_detected_clip(self, start_time, end_time):
        self.audio_data.seek(0)
        audio_segment = AudioSegment(
            data=self.audio_data.read(),
            sample_width=self.p.get_sample_size(self.format),
            frame_rate=self.rate,
            channels=self.channels
        )

        start_time -= self.audio_start
        end_time -= self.audio_start
        start_time *= 1000
        end_time *= 1000
        start_time -= self.start_padding

        clip_segment = audio_segment[start_time:end_time]
        segmented_clips = self.segment_clip(clip_segment)

        for i, seg in enumerate(segmented_clips):
            segment = AudioSegment(
                seg.tobytes(),
                sample_width=self.p.get_sample_size(self.format),
                frame_rate=self.rate,
                channels=self.channels
            )
            segment_filename = f"{self.output_dir}segment_{i}.wav"
            segment.export(segment_filename, format="wav")
            print(f"Saved segmented clip to {segment_filename}")

        self.detected_clip = True  # Set the flag to True when a clip is detected
        

    def segment_clip(self, clip_segment):
        audio_data = np.array(clip_segment.get_array_of_samples())
        plt.figure(figsize=(12, 4))
        plt.plot(audio_data, label='Audio Signal')
        plt.title('Audio Signal Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        voiced_segments = []
        silence_start = None
        voice_start = None
        is_voiced = False
        chunk_size = 32
        silence_threshold = 200 * self.rate / 1000
        padding_start = 100 * int(self.rate / 1000)
        padding_end = 200 * int(self.rate / 1000)

        segmentation_starts = []
        segmentation_ends = []

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunk_amplitude = 100 * np.abs(chunk).mean() / self.max_amplitude

            if chunk_amplitude >= self.threshold:
                silence_start = None

                if not is_voiced:
                    is_voiced = True
                    voice_start = i

            else:
                if is_voiced:
                    if silence_start is None:
                        silence_start = i

                    elif i - silence_start > silence_threshold:
                        voice_start -= padding_start
                        silence_start += padding_end

                        if voice_start < 0:
                            voice_start = 0
                        if silence_start > len(audio_data):
                            silence_start = len(audio_data)

                        print(f'Extracted segment [{voice_start}:{silence_start}]')
                        voiced_segments.append(audio_data[voice_start:silence_start])
                        segmentation_starts.append(voice_start)
                        segmentation_ends.append(silence_start)

                        silence_start = None
                        is_voiced = False

        if silence_start is not None and is_voiced:
            voiced_segments.append(audio_data[voice_start:])
            segmentation_starts.append(voice_start)
            segmentation_ends.append(len(audio_data))

        while len(voiced_segments) < 4:
            voiced_segments.append(AudioSegment.silent(duration=500))

        for point in segmentation_starts:
            plt.axvline(x=point, color='g', linestyle='--', label='Segment Start')

        for point in segmentation_ends:
            plt.axvline(x=point, color='r', linestyle='--', label='Segment End')

        plt.axhline(y=self.threshold * self.max_amplitude / 100, color='orange', linestyle=':', label='Silence Threshold')
        plt.grid()
        plt.show()
        plt.close()
        return voiced_segments[:4]

# Example usage
if __name__ == "__main__":
    while True:
        processor = AudioProcessor()
        processor.start_listening()
        while not processor.detected_clip:  # Keep running until a clip is detected
            time.sleep(1)
        processor.stop_listening()

