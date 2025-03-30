from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client
from flask import Flask, request, jsonify
from gradio_client import Client as GradioClient
import google.generativeai as genai
import json
import os
from flask_cors import CORS
from gemini import GiveGeminiOutput
from admin.markdown_converter import convert_to_markdown
from dotenv import load_dotenv
from admin.dataset_creator import create_dataset
from flask import Flask, request, jsonify, send_file
import threading
import cv2
import csv
import time
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os
import json
from mario.main import Core


app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize Gradio Client
gradio_client = GradioClient("yashMahajan/SanstheinAurSamvidhan")

# File to store chat history
history_file = "chat_history.json"

if os.path.exists(history_file):
    with open(history_file, "r") as f:
        chat_history = json.load(f)
else:
    chat_history = []

@app.route('/')
def hello_world():
    return 'Server is running!'

@app.route("/get_scenario", methods=["POST"])
def get_scenario():
    data = request.json
    message = data["message"]

    response = GiveGeminiOutput(message)

    return jsonify({"response": response})


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    message = data["message"]
    user_feedback = data.get("feedback", "")
    language = data["language"]

    # Perform the prediction using Gradio
    result = gradio_client.predict(
        message=message,
        system_message=f"You are a knowledgeable assistant specializing in the Constitution of India. Answer only questions related to the Constitution in {language}. If the answer does not contain relevant constitutional keywords, inform the user accordingly.",
        max_tokens=1500,
        temperature=0.7,
        top_p=0.95,
        api_name="/chat"
    )

    # Update chat history
    chat_history.append({
        "user_message": message,
        "response": result,
        "feedback": user_feedback
    })  

    # Save chat history
    with open(history_file, "w") as f:
        json.dump(chat_history, f, indent=4)

    return jsonify({"response": result, "conversation": chat_history})


@app.route("/markdown_converter", methods=["POST"])
def markdown_converter():
    # take the file from the request and convert it to markdown
    file = request.files['file']
    print(file.filename)
    if file:
        # save the file locally in temp folder create temp folder if not exists
        if not os.path.exists("temp"):
            os.makedirs("temp")
        file_path = os.path.join("temp", "input" + os.path.splitext(file.filename)[1])
        file.save(file_path)

        # convert the file to markdown
        if convert_to_markdown(file_path):
            # send stsus code 200
            return jsonify({"response": "Success"})
        
    return jsonify({"response": "Failed"})
        
@app.route("/get_markdown", methods=["GET"])
def get_markdown():
    # get the markdown file
    markdown_file = "./output/text.md"
    return send_file(markdown_file, as_attachment=True)

@app.route("/generate_dataset", methods=["POST"])
def generate_dataset():
    # take the markdown file from the request and generate the dataset
    file = request.files['file']
    print(file.filename)
    
    #save the file locally in temp folder create temp folder if not exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_path = os.path.join("temp", "input" + os.path.splitext(file.filename)[1])
    file.save(file_path)
    
    # generate the dataset
    create_dataset(file_path)
    # file is stored in the output folder dataset.zip
    
    return jsonify({"response": "Success"})

@app.route("/get_dataset", methods=["GET"])
def get_dataset():
    # get the dataset file
    dataset_file = "./output/dataset.zip"
    return send_file(dataset_file, as_attachment=True)

# main driver function
@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.get_json()
    text = data['text']

    # Initialize the API key and configure the Generative AI model
    genai.configure(api_key="AIzaSyDc-bZVat_laEHK3ultW6nqBxQ1pn4LRZM")

    # Set up the model with generation configuration and safety settings

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # Initialize and use the model to generate the story
    model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        safety_settings=safety_settings
    )

    convo = model.start_chat(history=[])
    convo.send_message(f"Give me a very short and simple story to understand this article: {text}")

    # Return the generated story
    return jsonify({"story": convo.last.text})

def run_game_1():
    oCore = Core()
    oCore.main_loop()
    return "Game 1 finished"

def run_game_2():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8)

    class MCQ():
        def __init__(self, data):
            self.question = data[0]
            self.choice1 = data[1]
            self.choice2 = data[2]
            self.choice3 = data[3]
            self.choice4 = data[4]
            self.answer = int(data[5])
            self.userAns = None

        def update(self, cursor, bboxs):
            for x, bbox in enumerate(bboxs):
                x1, y1, x2, y2 = bbox
                if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                    self.userAns = x + 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)

    pathCSV = "Mcqs.csv"
    with open(pathCSV, newline='\n') as f:
        reader = csv.reader(f)
        dataAll = list(reader)[1:]

    mcqList = [MCQ(q) for q in dataAll]

    qNo = 0
    qTotal = len(dataAll)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        if qNo < qTotal:
            mcq = mcqList[qNo]
            img, bbox = cvzone.putTextRect(img, mcq.question, [100, 100], 2, 2, offset=50, border=5)
            img, bbox1 = cvzone.putTextRect(img, mcq.choice1, [100, 250], 2, 2, offset=50, border=5)
            img, bbox2 = cvzone.putTextRect(img, mcq.choice2, [400, 250], 2, 2, offset=50, border=5)
            img, bbox3 = cvzone.putTextRect(img, mcq.choice3, [100, 400], 2, 2, offset=50, border=5)
            img, bbox4 = cvzone.putTextRect(img, mcq.choice4, [400, 400], 2, 2, offset=50, border=5)

            if hands:
                lmList = hands[0]['lmList']
                if len(lmList) >= 13:
                    cursor = lmList[8]
                    length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                else:
                    cursor = None
                    length, info = 0, ""
                if length < 60:
                    mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4])
                    if mcq.userAns is not None:
                        time.sleep(0.3)
                        qNo += 1
        else:
            score = sum(mcq.answer == mcq.userAns for mcq in mcqList)
            score = round((score / qTotal) * 100, 2)
            img, _ = cvzone.putTextRect(img, "Quiz is completed", [250, 300], 2, 2, offset=50, border=5)
            img, _ = cvzone.putTextRect(img, f'Your Score : {score}% ', [700, 300], 2, 2, offset=16, border=5)

        barValue = 150 + (950 // qTotal) * qNo
        cv2.rectangle(img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (150, 600), (1100, 650), (255, 0, 255), 5)
        img, _ = cvzone.putTextRect(img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 2, 2, offset=16, border=5)

        cv2.imshow("Img", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Game 2 finished"

@app.route('/start-game-1', methods=['GET'])
def start_game_1():
    game_thread = threading.Thread(target=run_game_1)
    game_thread.start()
    return jsonify({"message": "Game 1 started!"})

@app.route('/start-game-2', methods=['GET'])
def start_game_2():
    quiz_thread = threading.Thread(target=run_game_2)
    quiz_thread.start()
    return jsonify({"message": "Quiz game started!"})

@app.route('/game-status', methods=['GET'])
def game_status():
    status = request.args.get('status', 'unknown')
    return jsonify({"status": status})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
