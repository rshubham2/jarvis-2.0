import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pyttsx3
import speech_recognition as sr


class SpeechThread(QThread):
    textChanged = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()

    def run(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio)
            self.textChanged.emit(text)
        except sr.UnknownValueError:
            self.textChanged.emit("Sorry, I didn't catch that.")
        except sr.RequestError:
            self.textChanged.emit("Sorry, there was an error processing your request.")


class JarvisGUI(QMainWindow):
    def __init__(self, process_func):
        super().__init__()
        self.process_func = process_func
        self.initUI()
        self.engine = pyttsx3.init()
        self.speech_thread = SpeechThread()
        self.speech_thread.textChanged.connect(self.onSpeechRecognized)

    def initUI(self):
        self.setWindowTitle('Jarvis Virtual Assistant')
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('jarvis_icon.png'))  # Make sure to have this icon file

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setFixedHeight(50)
        input_layout.addWidget(self.input_field)

        send_button = QPushButton('Send')
        send_button.clicked.connect(self.onSendClicked)
        input_layout.addWidget(send_button)

        main_layout.addLayout(input_layout)

        # Control buttons
        button_layout = QHBoxLayout()

        mic_button = QPushButton('🎤')
        mic_button.clicked.connect(self.onMicClicked)
        button_layout.addWidget(mic_button)

        wake_button = QPushButton('Wake Up')
        wake_button.clicked.connect(self.onWakeUpClicked)
        button_layout.addWidget(wake_button)

        sleep_button = QPushButton('Go to Sleep')
        sleep_button.clicked.connect(self.onSleepClicked)
        button_layout.addWidget(sleep_button)

        main_layout.addLayout(button_layout)

        # Status bar
        self.statusBar().showMessage('Jarvis is ready')

    def onSendClicked(self):
        user_input = self.input_field.toPlainText()
        self.chat_display.append(f"You: {user_input}")
        self.input_field.clear()
        self.processCommand(user_input)

    def onMicClicked(self):
        self.statusBar().showMessage('Listening...')
        self.speech_thread.start()

    def onSpeechRecognized(self, text):
        self.statusBar().showMessage('Jarvis is ready')
        self.chat_display.append(f"You: {text}")
        self.processCommand(text)

    def onWakeUpClicked(self):
        self.chat_display.append("Jarvis: Waking up, sir. How may I assist you?")
        self.speak("Waking up, sir. How may I assist you?")

    def onSleepClicked(self):
        self.chat_display.append("Jarvis: Going to sleep mode. Wake me up when you need me.")
        self.speak("Going to sleep mode. Wake me up when you need me.")

    def processCommand(self, command):
        response = self.process_func(command)
        self.chat_display.append(f"Jarvis: {response}")
        self.speak(response)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


def run_gui(process_func):
    app = QApplication(sys.argv)
    ex = JarvisGUI(process_func)
    ex.show()
    sys.exit(app.exec_())