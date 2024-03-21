from interviewer import solutions,read_squad
import speech_recognition as sr
from speaker import Speak
import spacy
def Listen():

    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold=1
        audio=r.listen(source,0,8)
    try:
        print("Recognizing...")
        query=r.recognize_google(audio,language="en")
    except:
        return ""
    
    query=str(query).lower()
    return query
def interview():
        contexts, questions, answers = read_squad('C:/Users/piyush gupta/OneDrive/Desktop/studyBuddy/Server/dataset/Train-v2.0.json') 
        flag=0
        Speak("hello piyush get ready for your interview")
        abcd = Listen()

        m=0

        dict ={
            'user_answer': [],
            'question': [],
            'accuracy': [],
            'similarity': []

        }
        template="you are an interviewer you asked a question and candiate answers,give some feedback to the candidate for improvement the question answer and accuracy is given as follows give dtailed questionwise feedback"
        if abcd != "":
                while flag!=2:
                        avg = 0
                        Speak(questions[m])
                        q = Listen()
                        print(q)
                        nlp = spacy.load("en_core_web_md")
                        ans = solutions(m,m)
                        ans = ans['answer']
                        print(ans)
                        q1 = nlp(q)
                        ans1 = nlp(ans)
                        avg = avg + q1.similarity(ans1)
                        tmp = avg
                        q_verb = "".join([token.lemma_ for token in q1 if token.pos_ =="VERB"]) 
                        q_noun = "".join([token.lemma_ for token in q1 if token.pos_ =="NOUN"])
                        q_adj = "".join([token.lemma_ for token in q1 if token.pos_ =="ADJ"])
                        ans_verb = "".join([token.lemma_ for token in ans1 if token.pos_ =="VERB"]) 
                        ans_noun = "".join([token.lemma_ for token in ans1 if token.pos_ =="NOUN"])
                        ans_adj = "".join([token.lemma_ for token in ans1 if token.pos_ =="ADJ"])
                        if len(q_verb) !=0 and len(ans_verb) !=0 : 
                            avg = avg + nlp(q_verb).similarity(nlp(ans_verb))
                        if len(q_noun) !=0 and len(ans_noun) !=0 :
                            avg = avg +  nlp(q_adj).similarity(nlp(ans_adj))
                        if len(q_adj) !=0 and len(ans_adj) !=0 :
                            avg = avg + nlp(q_noun).similarity(nlp(ans_noun))
                        avg = avg/4
                        avg = avg + 0.2

                        dict['user_answer'].append(q)
                        dict['question'].append(questions[m])
                        dict['accuracy'].append(avg)
                        dict['similarity'].append(tmp)
                        m=m+1
                        flag=flag+1
        else:
            print("sorry for intruption")
        
        for i in range(0,2):
             template +=("question"+str(i))
             template += dict["question"][i]
             template +=("answer "+str(i))
             template += dict["user_answer"][i]
             template +=("accuracy "+str(i))
             template += str(dict["accuracy"][i])
             
        
        print(template)
        template+="most importantly remember to break the result into  2-3 stanzas"
        template+="                                                                                                                                                                  "
        return template
        