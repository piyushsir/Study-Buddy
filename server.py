from flask import Flask,jsonify, request, Response, render_template
from psycho_test import judgement
from flask_cors import CORS
from listener import interview 
app = Flask(__name__)
CORS(app)
global q1
@app.route("/api/home",methods = ['GET', 'POST'])
def return_home():
    result = interview()
    #print(result['question'])
    return jsonify({'data':result})
@app.route("/test", methods = ["GET", "POST"])
def test():
    global q1
    if request.method == "POST":
        q1 = request.form.get('question1')
        q2 = request.form.get('question2')
        q3 = request.form.get('question3')
        q4 = request.form.get('question4')
        q5 = request.form.get('question5')
        
        text=""
        text+=q1
        text+=q2
        text+=q3
        text+=q4
        text+=q5
        print(text)
        res=""
        if text!="":
          res = judgement(text)
        #   res = "Lorem ipsum dolor sit amet consectetur adipisicing elit. Fugit minus fuga veniam eveniet. Sunt quasi, animi officia delectus tempora consequatur ullam corrupti dolore quidem aperiam quod adipisci rerum omnis enim unde quas quia eligendi non? In exercitationem ipsa corrupti mollitia? Officiis quae eligendi expedita voluptas impedit doloremque ex sunt cupiditate corporis! Sapiente iure odio consequuntur eos voluptatibus beatae placeat quod iusto aut fugiat, sunt explicabo repudiandae laudantium similique quo quasi pariatur nam itaque expedita ipsam reprehenderit quibusdam esse! Velit fuga neque tempora nulla consequatur qui! Voluptates enim consequatur officia tenetur corporis accusantium numquam nisi magnam a quasi labore, nobis quis quidem assumenda, neque quaerat delectus, commodi iste nulla porro. Saepe, sunt fugiat harum repellendus consequatur enim quisquam repellat maxime veritatis! Numquam reprehenderit aliquam commodi, explicabo est eligendi minima nulla impedit veritatis. Tempore ipsum vero maiores. Nam quo eligendi ipsa, fuga commodi vel maxime soluta quidem repudiandae fugiat ea totam vitae. Totam consequuntur odio, amet tenetur, enim minima, in sed deleniti debitis voluptatibus quisquam consectetur accusantium vitae quaerat quasi explicabo quam! Fuga natus soluta eveniet, sunt in hic dolorum recusandae corporis magni! Cupiditate illum quis nostrum culpa, et facilis nihil unde voluptates ut asperiores molestias sequi quo incidunt maxime officia dolorum."
          return render_template("form.html", sub = None, res = res)
          with open("C:/Users/piyush gupta/OneDrive/Desktop/studyBuddy/Server/characteristic.txt","a") as f:
              f.write(res)
          
          
        return render_template("form.html", sub = "SUBMIT", res = None)
    return render_template("form.html", sub = None, res = None)

if __name__ == "__main__":
    app.run(debug=True,port=8080)