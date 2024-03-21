import json
from transformers import BertForQuestionAnswering

def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                
                for answer in qa['answers']:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad('C:/Users/piyush gupta/OneDrive/Desktop/studyBuddy/Server/dataset/Train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('C:/Users/piyush gupta/OneDrive/Desktop/studyBuddy/Server/dataset/Dev-v2.0.json')


from transformers import DistilBertForQuestionAnswering
def solutions(question,context):
        from transformers import BertForQuestionAnswering
        modelin =BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")


        from transformers import AutoTokenizer
        tokenizerr = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        tokenizerr.encode(train_questions[question], truncation=True, padding=True)
        from transformers import pipeline
        nlp = pipeline('question-answering',model=modelin, tokenizer=tokenizerr)
        solu = nlp({
            "question": train_questions[question],
            "context": train_contexts[context]
        })

        return solu





if __name__ == "__main__":
    pass


