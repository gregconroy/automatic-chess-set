import pyaudio
import numpy as np
from pydub import AudioSegment
import time
import io
import os

class DataCollector:
    MAX_AMPLITUDE = 2**15 # 16-bit audio
    CHANNELS = 1
    SAMPLING_RATE = 44100 # Hz
    FORMAT = pyaudio.paInt16
    BUFFER_SIZE = 64 # frames
    CLIP_ACTIVATION_THRESHOLD = 2 # percent
    CLIP_MIN_SILENCE_DURATION = 200 # ms
    CLIP_MIN_DURATION = 50 # ms
    CLIP_START_PADDING = 100 # ms
    CLIP_END_PADDING = 200 # ms

    SEGMENT_BUFFER_SIZE = 32 # frames
    SEGMENT_ACTIVATION_THRESHOLD = 4 # percent
    SEGMENT_MIN_SILENCE_DURATION = 200 # ms
    SEGMENT_START_PADDING = 50 # ms
    SEGMENT_END_PADDING = 200 # ms

    def __init__(self, output_directory):
        self.OUTPUT_DIR = output_directory

        self.py_audio = pyaudio.PyAudio()
        self.audio_data = io.BytesIO()
        self.stream = None
        
        # clip extraction variables
        self.voice_active = False
        self.silence_start_index = None
        self.clip_start_index = None
        self.clip_end_index = None

        self.recording = False
        self.current_take = None
        self.name = None

        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

    def start_listening(self):
        """ start the audio stream """
        self.stream = self.py_audio.open(input=True,
                                         format=self.FORMAT,
                                         channels=self.CHANNELS,
                                         rate=self.SAMPLING_RATE,
                                         frames_per_buffer=self.BUFFER_SIZE,
                                         stream_callback=self.__handle_audio_chunk)     
        self.stream.start_stream()
        print('Listening...')
        

    def stop_listening(self):
        """ stop the audio stream """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.py_audio.terminate()
            print("Stopped listening.")


    def collect_data(self):
        if not self.name:
            self.name = input("Name: ").strip()
            self.category = input("Category: ").strip()
        self.piece = input("Piece: ").strip()
        take_range = input("Take range: ").strip()
        self.take_min, self.take_max = map(int, take_range.split('-'))
        self.current_take = self.take_min
        self.recording = True


    def __handle_audio_chunk(self, in_data, frame_count, time_info, flag):
        """ process chunk of audio frames """
        if not self.recording:
            return in_data, pyaudio.paContinue    

        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        self.audio_data.write(in_data)

        if self.__is_voice_activity(audio_chunk, self.CLIP_ACTIVATION_THRESHOLD):
            # there is voice activity
            if self.voice_active:
                # voice already active
                self.silence_start_index = self.__get_current_audio_index()
            else:
                # voice was inactive
                print("Voice activity detected")

                self.voice_active = True
                self.silence_start_index = None
                self.clip_start_index = self.__get_current_audio_index()
        else:
            # there is no voice activity
            if self.voice_active:
                # voice was active
                if self.silence_start_index is None:
                    # set the silence start index if first time silence detected
                    self.silence_start_index = self.__get_current_audio_index()

                if self.__min_silence_duration_surpassed(self.__get_current_audio_index(), self.silence_start_index, self.CLIP_MIN_SILENCE_DURATION):
                    # legnth of silence deems no voice activity
                    print("Voice activity stopped")

                    self.clip_end_index = self.silence_start_index
                    clip_duration = self.clip_end_index - self.clip_start_index

                    print(f"Clip duration: {self.__frames_to_ms(clip_duration)}ms")

                    if self.__is_valid_clip(clip_duration):
                        # clip length is long enough
                        print("Clip length is valid")
                        self.__process_clip(self.clip_start_index, self.clip_end_index)

                    # reset all variables
                    self.voice_active = False
                    self.clip_start_index = None
                    self.clip_end_index = None
                    self.silence_start_index = None   

        return in_data, pyaudio.paContinue     


    def __process_clip(self, start_index, end_index):
        """ extract the spoken move clip from the audio buffer for further processing """
        start_index -= self.__ms_to_frames(self.CLIP_START_PADDING)
        end_index += self.__ms_to_frames(self.CLIP_END_PADDING)

        self.audio_data.seek(0)
        audio_buffer = self.audio_data.read()

        audio_buffer_data = np.frombuffer(audio_buffer, dtype=np.int16)
        clip_data = audio_buffer_data[start_index:end_index]

        self.__save_clip(clip_data)

        self.audio_data = io.BytesIO()


    def __save_clip(self, clip_data):
        # Convert the clip data to a pydub AudioSegment
        clip_segment = AudioSegment(
            clip_data.tobytes(), 
            frame_rate=self.SAMPLING_RATE, 
            sample_width=self.py_audio.get_sample_size(self.FORMAT), 
            channels=self.CHANNELS
        )

        # Save the clip as a .wav file
        output_filename = f"{self.OUTPUT_DIR}{self.category}/{self.name}_{self.piece}_{self.current_take}.wav"
        clip_segment.export(output_filename, format="wav")
        print(f"Saved clip to {output_filename}")

        self.current_take += 1
        if (self.current_take > self.take_max):
            self.recording = False
            print('Recording concluded')
            self.collect_data()


            
    def __get_current_audio_index(self):
        """ get the current index of streamed audio data """
        current_position = self.audio_data.tell()
        self.audio_data.seek(0)
        audio_bytes = self.audio_data.read()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        self.audio_data.seek(current_position)

        return len(audio_array)


    def __get_amplitude(self, audio_chunk):
        """ normalise amplitude to a percentage """
        return 100 * (np.abs(audio_chunk).mean() / self.MAX_AMPLITUDE)
    

    def __is_voice_activity(self, audio_chunk, activation_threshold):
        """ determine if amplitude is greater than activation threshold """
        amplitude = self.__get_amplitude(audio_chunk)
        return amplitude >= activation_threshold
    

    def __is_valid_clip(self, clip_duration):
        """ determine if clip length is long enough for a spoken move """
        return clip_duration >= self.__ms_to_frames(self.CLIP_MIN_DURATION)


    def __min_silence_duration_surpassed(self, current, start, threshold):
        """ determine if duration of silence surpasses the threshold """
        return current - start >= self.__ms_to_frames(threshold)
        

    def __ms_to_frames(self, ms):
        """ convert milliseconds to audio frames """
        return int(self.SAMPLING_RATE * (ms / 1000))
    

    def __frames_to_ms(self, frames):
        """ convert audio frames to milliseconds """
        return 1000 * (frames / self.SAMPLING_RATE)


if __name__ == "__main__":
    data_collector = DataCollector(output_directory='./Training Data/')

    try:
        data_collector.start_listening()
        data_collector.collect_data()
        while True:
            time.sleep(1)  # Keep the program running
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping...")
        data_collector.stop_listening()

