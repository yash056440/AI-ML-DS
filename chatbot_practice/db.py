"""
db.py — SQLite data layer for the Student Result Management app.

Two tables:
  students(id, name, roll_no, gender, std, created_at)
  marks(id, student_id, subject, marks, max_marks)

Percentage / grade are always computed on the fly from `marks`,
never stored, so they can never go stale after an edit.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "students.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT,
            gender TEXT,
            std TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            marks REAL NOT NULL,
            max_marks REAL NOT NULL DEFAULT 100,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


def grade_for(pct):
    if pct >= 90:
        return "A+"
    if pct >= 75:
        return "A"
    if pct >= 60:
        return "B"
    if pct >= 40:
        return "C"
    return "F"


def subject_level(pct_of_subject):
    """Qualitative bucket used for chatbot feedback sentences."""
    if pct_of_subject >= 75:
        return "good"
    if pct_of_subject >= 40:
        return "average"
    return "weak"


def _student_row_to_dict(conn, row):
    marks_rows = conn.execute(
        "SELECT subject, marks, max_marks FROM marks WHERE student_id = ? ORDER BY subject",
        (row["id"],),
    ).fetchall()
    subjects = {
        m["subject"]: {"marks": m["marks"], "max_marks": m["max_marks"]}
        for m in marks_rows
    }
    total = sum(m["marks"] for m in marks_rows)
    max_total = sum(m["max_marks"] for m in marks_rows)
    pct = round((total / max_total) * 100, 2) if max_total else 0.0
    return {
        "id": row["id"],
        "name": row["name"],
        "roll_no": row["roll_no"],
        "gender": row["gender"],
        "std": row["std"],
        "subjects": subjects,
        "total": total,
        "max_total": max_total,
        "percentage": pct,
        "grade": grade_for(pct),
    }


def list_students():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    result = [_student_row_to_dict(conn, r) for r in rows]
    conn.close()
    return result


def get_student(student_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    if not row:
        conn.close()
        return None
    d = _student_row_to_dict(conn, row)
    conn.close()
    return d


def find_students_by_name(name_fragment):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM students WHERE LOWER(name) LIKE ?",
        (f"%{name_fragment.lower()}%",),
    ).fetchall()
    result = [_student_row_to_dict(conn, r) for r in rows]
    conn.close()
    return result


def add_student(name, roll_no, gender, std, subjects):
    """subjects: dict of {subject_name: {"marks": x, "max_marks": y}}"""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO students (name, roll_no, gender, std) VALUES (?, ?, ?, ?)",
        (name, roll_no, gender, std),
    )
    student_id = cur.lastrowid
    for subject, m in subjects.items():
        conn.execute(
            "INSERT INTO marks (student_id, subject, marks, max_marks) VALUES (?, ?, ?, ?)",
            (student_id, subject, m["marks"], m.get("max_marks", 100)),
        )
    conn.commit()
    conn.close()
    return student_id


def update_student(student_id, name, roll_no, gender, std, subjects):
    conn = get_conn()
    conn.execute(
        "UPDATE students SET name=?, roll_no=?, gender=?, std=? WHERE id=?",
        (name, roll_no, gender, std, student_id),
    )
    conn.execute("DELETE FROM marks WHERE student_id = ?", (student_id,))
    for subject, m in subjects.items():
        conn.execute(
            "INSERT INTO marks (student_id, subject, marks, max_marks) VALUES (?, ?, ?, ?)",
            (student_id, subject, m["marks"], m.get("max_marks", 100)),
        )
    conn.commit()
    conn.close()


def delete_student(student_id):
    conn = get_conn()
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()


def all_subjects():
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT subject FROM marks ORDER BY subject").fetchall()
    conn.close()
    return [r["subject"] for r in rows]
