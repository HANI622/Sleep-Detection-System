from flask import Flask, render_template, Response, jsonify
from detector import generate_frames
from shared import dashboard_data

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/data")
def data():
    return jsonify(dashboard_data)

if __name__ == "__main__":
    app.run(debug=True)