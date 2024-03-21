from psycho import psycho_test
import openai

openai.api_key = "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"


def judgement(text):
    
    template="you are given the nature of a person ,as a psychologist suggest him how can he be more productive in first paragraph and suggest him a good timetable for him in the second paragraph the person is identified as "
    res = psycho_test(text)
    template+=res
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": template}])
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
