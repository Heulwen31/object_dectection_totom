import os
from gtts import gTTS
from playsound import playsound


def text_to_mp3(text, filename, lang='en', slow=False):
    """
    Convert text to speech and save it as an MP3 file.

    Parameters:
    - text: The text to convert to speech.
    - filename: The name of the output MP3 file.
    - lang: The language to use for the speech (default is English).
    - slow: Whether to speak slowly (default is False).
    """
    try:
        # Initialize gTTS object
        tts = gTTS(text=text, lang=lang, slow=slow)
        # Save the speech to an MP3 file
        tts.save(filename)
        print(f"Saved MP3 file as {filename}")
        # Play the MP3 file
        # playsound(filename)
        # print("Sucess read file audio")
        # # remove
        os.remove(filename)
        print("Success remove audio")
    except Exception as e:
        print(f"An error occurred: {e}")