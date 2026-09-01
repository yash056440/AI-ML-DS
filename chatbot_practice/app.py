from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

import os
import db
import chatbot

# Load the API key from config.env (a plain file, no leading dot,
# so it always shows up normally in Windows Explorer / Finder).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "config.env"))

app = Flask(__name__)
db.init_db()


@app.route("/")
def index():
    return render_template("index.html")


# ---------- Student CRUD ----------

@app.route("/api/students", methods=["GET"])
def get_students():
    return jsonify(db.list_students())


@app.route("/api/students", methods=["POST"])
def create_student():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    subjects = data.get("subjects") or {}
    student_id = db.add_student(
        name=name,
        roll_no=data.get("roll_no"),
        gender=data.get("gender"),
        std=data.get("std"),
        subjects=subjects,
    )
    return jsonify(db.get_student(student_id)), 201


@app.route("/api/students/<int:student_id>", methods=["PUT"])
def update_student(student_id):
    if not db.get_student(student_id):
        return jsonify({"error": "Student not found"}), 404
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    subjects = data.get("subjects") or {}
    db.update_student(
        student_id,
        name=name,
        roll_no=data.get("roll_no"),
        gender=data.get("gender"),
        std=data.get("std"),
        subjects=subjects,
    )
    return jsonify(db.get_student(student_id))


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    if not db.get_student(student_id):
        return jsonify({"error": "Student not found"}), 404
    db.delete_student(student_id)
    return jsonify({"ok": True})


# ---------- Chatbot ----------

@app.route("/api/chatbot", methods=["POST"])
def chatbot_query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        reply, intent_obj = chatbot.answer(question)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500
    return jsonify({"reply": reply, "intent": intent_obj})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
