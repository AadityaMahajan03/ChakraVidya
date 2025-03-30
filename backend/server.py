from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
import bcrypt
from balVidya.home.fingerCount import finger_counter
from balVidya.home.senseOrgan import sense_organ_labeling
from balVidya.home.manners import manners
from balVidya.home.interactiveReader import interactiveReader
from flask import Flask, render_template, request, jsonify, send_file
from gradio_client import Client
from flask import Flask, request, jsonify
from gradio_client import Client as GradioClient
import google.generativeai as genai
import json
import os
from flask_cors import CORS
from dotenv import load_dotenv
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
CORS(app)  # Allow cross-origin requests

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/balvidya"
mongo = PyMongo(app)
users = mongo.db.users  # Users collection

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    full_name = data.get("fullName")
    email = data.get("email")
    password = data.get("password")
    user_type = data.get("userType")
    grade = data.get("grade", "")

    if users.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    user_data = {
        "fullName": full_name,
        "email": email,
        "password": hashed_pw,
        "userType": user_type,
        "grade": grade
    }

    users.insert_one(user_data)
    return jsonify({"message": "User registered successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = users.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"message": "Invalid credentials"}), 401

    user.pop("password")  # Remove password before sending response
    return jsonify({"message": "Login successful", "user": user}), 200

    
@app.route("/get-username", methods=["POST"])
def get_username():
    data = request.json
    email = data.get("email")

    user = users.find_one({"email": email}, {"fullName": 1, "_id": 0})
    if not user:
        return jsonify({"message": "User not found"}), 404

    return jsonify({"fullName": user["fullName"]}), 200

@app.route("/balVidya/home/math", methods=["GET"])
def execute_finger_counter():
    # execute the finger_counter function
    finger_counter()
    # if the finger counter is success it is return the success message
    start_game_1()
    return jsonify({"message": "Finger counter executed successfully"}), 200

@app.route("/balVidya/home/sense-organ", methods=["GET"])
def execute_sense_organ():
    sense_organ_labeling()
    start_game_1()
    return jsonify({"message": "Sense organ labeling executed successfully"}), 200

@app.route("/get-user-object",methods=["POST"])
def get_user_object():
    data = request.json
    email = data.get("email")
    user = users.find_one({"email": email})
    if not user:
        return jsonify({"message": "User not found"}), 404
    return jsonify({"user": user}), 200

@app.route("/balVidya/home/manners",methods=["GET"])
def execute_manners():
    manners()
    start_game_1()
    return jsonify({"message": "Sense organ labeling executed successfully"}), 200

@app.route("/balVidya/home/interactiveReader",methods=["GET"])
def execute_interactiveReader():
    interactiveReader()
    start_game_1()
    return jsonify({"message": "Interactive Reader Successful"}), 200


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


if __name__ == "__main__":
    app.run(debug=True)


