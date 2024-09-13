import os
import random
import webbrowser
import pyttsx3
import speech_recognition as sr
import speedtest
from pygame import mixer
from mtranslate import translate
import sys
import torch
import pyautogui
from plyer import notification

# Add the directory containing the 'jarvis' folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanoGPT.train import (
    load_models_and_tokenizers,
    load_emotions,
    predict_emotion,
    generate_response_from_example,
    retrieve_similar_example,
    load_dataset
)

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)
rate = engine.setProperty("rate", 170)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def translate_hindi_to_english(text):
    return translate(text, "en-us")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        r.pause_threshold = 1
        r.energy_threshold = 300
        audio = r.listen(source, timeout=5, phrase_time_limit=5)

    try:
        print("Understanding..")
        query = r.recognize_google(audio, language='en-in')
        print(f"You Said: {query}\n")
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Could you please repeat?")
        return "None"
    except sr.RequestError:
        print("Sorry, my speech service is down. Please check your internet connection.")
        return "None"

def process_command(query, emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model, device, emotions):
    # Predict emotion
    emotion, confidence = predict_emotion(emotion_model, emotion_tokenizer, query, device, emotions)
    print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

    # Generate response
    situations, responses = load_dataset()
    similar_situation, retrieved_response, similarity_score = retrieve_similar_example(query, situations, responses, similarity_model)
    response = generate_response_from_example(similar_situation, retrieved_response, response_model, response_tokenizer, query, device, emotion)

    print(f"Generated response: {response}")
    speak(response)

    # Process commands
    if "google" in query:
        from SearchNow import searchGoogle

        searchGoogle(query)

    elif "youtube" in query:
        from SearchNow import searchYoutube

        searchYoutube(query)
    elif "wikipedia" in query:
        from SearchNow import searchWikipedia

        searchWikipedia(query)
    elif "set an alarm" in query:
        print("input time example:- 10 and 10 and 10")
        speak("Set the time")
        a = input("Please tell the time :- ")
        alarm(a)
        speak("Done,sir")
    elif "pause" in query:
        pyautogui.press("k")
        speak("video paused")
    elif "play" in query:
        pyautogui.press("k")
        speak("video played")
    elif "mute" in query:
        pyautogui.press("m")
        speak("video muted")

    elif "volume up" in query:
        from keyboard import volumeup

        speak("Turning volume up,sir")
        volumeup()
    elif "volume down" in query:
        from keyboard import volumedown

        speak("Turning volume down, sir")
        volumedown()
    elif "remember that" in query:
        rememberMessage = query.replace("remember that", "")
        rememberMessage = query.replace("jarvis", "")
        speak("You told me to remember that" + rememberMessage)
        remember = open("Remember.txt", "a")
        remember.write(rememberMessage)
        remember.close()
    elif "what do you remember" in query:
        remember = open("Remember.txt", "r")
        speak("You told me to remember that" + remember.read())
    elif "tired" in query:
        speak("Playing your favourite songs, sir")
        a = (1)  # You can choose any number of songs (I have only choosen 3)
        b = random.choice(a)
        if b == 1:
            webbrowser.open("https://www.youtube.com/watch?v=9bZkp7q19f0")  # Here put the link of your video)
    elif "news" in query:
        from NewsRead import latestnews

        latestnews()
    elif "calculate" in query:
        from Calculatenumbers import WolfRamAlpha
        from Calculatenumbers import Calc

        query = query.replace("calculate", "")
        query = query.replace("jarvis", "")
        Calc(query)
    elif "whatsapp" in query:
        from Whatsapp import sendMessage

        sendMessage()
    elif "shutdown the system" in query:
        speak("Are You sure you want to shutdown")
        shutdown = input("Do you wish to shutdown your computer? (yes/no)")
        if shutdown == "yes":
            os.system("shutdown /s /t 1")

        elif shutdown == "no":
            pass
    elif "change password" in query:
        speak("What's the new password")
        new_pw = input("Enter the new password\n")
        new_password = open("password.txt", "w")
        new_password.write(new_pw)
        new_password.close()
        speak("Done sir")
        speak(f"Your new password is{new_pw}")
    elif "schedule my day" in query:
        tasks = []  # Empty list
        speak("Do you want to clear old tasks (Plz speak YES or NO)")
        query = takeCommand().lower()
        if "yes" in query:
            file = open("tasks.txt", "w")
            file.write(f"")
            file.close()
            no_tasks = int(input("Enter the no. of tasks :- "))
            i = 0
            for i in range(no_tasks):
                tasks.append(input("Enter the task :- "))
                file = open("tasks.txt", "a")
                file.write(f"{i}. {tasks[i]}\n")
                file.close()
        elif "no" in query:
            i = 0
            no_tasks = int(input("Enter the no. of tasks :- "))
            for i in range(no_tasks):
                tasks.append(input("Enter the task :- "))
                file = open("tasks.txt", "a")
                file.write(f"{i}. {tasks[i]}\n")
                file.close()
    elif "show my schedule" in query:
        file = open("tasks.txt", "r")
        content = file.read()
        file.close()
        mixer.init()
        mixer.music.load("notification.mp3")
        mixer.music.play()
        notification.notify(
            title="My schedule :-",
            message=content,
            timeout=15
        )
    elif "open" in query:  # EASY METHOD
        query = query.replace("open", "")
        query = query.replace("jarvis", "")
        pyautogui.press("super")
        pyautogui.typewrite(query)
        pyautogui.sleep(2)
        pyautogui.press("enter")
    elif "internet speed" in query:
        wifi = speedtest.Speedtest()
        upload_net = wifi.upload() / 1048576  # Megabyte = 1024*1024 Bytes
        download_net = wifi.download() / 1048576
        print("Wifi Upload Speed is", upload_net)
        print("Wifi download speed is ", download_net)
        speak(f"Wifi download speed is {download_net}")
        speak(f"Wifi Upload speed is {upload_net}")
    elif "ipl score" in query:
        from plyer import notification  # pip install plyer
        import requests  # pip install requests
        from bs4 import BeautifulSoup  # pip install bs4

        url = "https://www.cricbuzz.com/"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        team1 = soup.find_all(class_="cb-ovr-flo cb-hmscg-tm-nm")[0].get_text()
        team2 = soup.find_all(class_="cb-ovr-flo cb-hmscg-tm-nm")[1].get_text()
        team1_score = soup.find_all(class_="cb-ovr-flo")[8].get_text()
        team2_score = soup.find_all(class_="cb-ovr-flo")[10].get_text()

        a = print(f"{team1} : {team1_score}")
        b = print(f"{team2} : {team2_score}")

        notification.notify(
            title="IPL SCORE :- ",
            message=f"{team1} : {team1_score}\n {team2} : {team2_score}",
            timeout=15
        )
    elif "play a game" in query:
        from game import game_play

        game_play()
    elif "screenshot" in query:
        import pyautogui  # pip install pyautogui

        im = pyautogui.screenshot()
        im.save("ss.jpg")
    elif "click my photo" in query:
        pyautogui.press("super")
        pyautogui.typewrite("camera")
        pyautogui.press("enter")
        pyautogui.sleep(2)
        speak("SMILE")
        pyautogui.press("enter")
    elif "focus mode" in query:
        a = int(input("Are you sure that you want to enter focus mode :- [1 for YES / 2 for NO "))
        if (a == 1):
            speak("Entering the focus mode....")
            os.startfile("C:\\Users\\adity\\OneDrive\\Desktop\\JARVIS 3.0\\FocusMode.py")
            exit()
        else:
            pass
    elif "show my focus" in query:
        from FocusGraph import focus_graph

        focus_graph()
    elif "translate" in query:
        from Translator import translategl

        query = query.replace("jarvis", "")
        query = query.replace("translate", "")
        translategl(query)
    elif "open ai model" in query:
        speak("Certainly, I'll use the AI model to generate a response.")
        user_input = takeCommand().lower()

        # Predict emotion
        emotion, confidence = predict_emotion(emotion_model, emotion_tokenizer, user_input, device,
                                              emotions)
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

        # Generate response
        situations, responses = load_dataset()
        similar_situation, retrieved_response, similarity_score = retrieve_similar_example(
            user_input, situations, responses, similarity_model)
        ai_response = generate_response_from_example(similar_situation, retrieved_response,
                                                     response_model, response_tokenizer, user_input,
                                                     device, emotion)

        speak(f"The AI model says: {ai_response}")

def alarm(query):
    timehere = open("Alarmtext.txt", "a")
    timehere.write(query)
    timehere.close()
    os.startfile("alarm.py")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model = load_models_and_tokenizers()
    emotions = load_emotions()

    # Move models to device
    emotion_model.to(device)
    response_model.to(device)
    similarity_model.to(device)

    speak("Hello! I'm Jarvis. How can I assist you today?")

    while True:
        query = takeCommand()
        if query == "None":
            continue

        if "wake up" in query:
            speak("I'm awake and ready to assist you.")
            continue

        if "go to sleep" in query:
            speak("Goodbye! You can wake me up anytime you need assistance.")
            break

        process_command(query, emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model, device, emotions)



if __name__ == "__main__":
    main()


