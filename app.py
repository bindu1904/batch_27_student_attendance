
from flask import Flask, render_template, request
from model import predict_attendance

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "class_time": int(request.form["class_time"]),
        "day": int(request.form["day"]),
        "subject_difficulty": int(request.form["subject_difficulty"]),
        "teaching_style": int(request.form["teaching_style"]),
        "weather": int(request.form["weather"]),
        "student_interest": int(request.form["student_interest"]),
        "exam_near": int(request.form["exam_near"])
    }

    feedback = request.form["feedback"]
    result = predict_attendance(data, feedback)

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
