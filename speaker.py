
import pyttsx3

def Speak(Text):
    eng=pyttsx3.init("sapi5")
    voices=eng.getProperty('voices')
    eng.setProperty('voices',voices[1].id)
    eng.setProperty('rate',155)
    print("")
    print(f"You : {Text}")
    print("")
    eng.say(Text)
    eng.runAndWait()


