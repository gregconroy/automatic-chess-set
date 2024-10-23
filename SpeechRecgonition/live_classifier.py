from live_segmentation import LiveSpeechProcessor
from move_classifier import MoveClassifier
import time


def classify_move(audio_segments):
    print('Classifying move')
    predictions = move_classifier.classify(audio_segments)

    move = ''

    for category, prediction in predictions.items():
        print(category)
        print(prediction)
        move += prediction[-1] + ' '

    print(f'Final move: {move}')


move_classifier = MoveClassifier()
speech_processor = LiveSpeechProcessor(classify_callback=classify_move)

try:
    speech_processor.start_listening()
    while True:
        time.sleep(1)  # Keep the program running
except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Stopping...")
    speech_processor.stop_listening()
