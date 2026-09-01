"""
chatbot.py — 3-stage pipeline described by the user:

  Stage 1: user question -> Groq (Llama 3.3-70b) -> structured "intent" JSON
           (ONLY the question text + the list of known subjects/std values
            is sent to the API — never any student's personal data)
  Stage 2: intent JSON -> executed locally against SQLite (db.py)
  Stage 3: local result -> natural-language sentence (template-based,
           no API call needed, so results are fast and data never
           leaves the server at this stage either)
"""

import os
import json
import re
from groq import Groq

import db

GROQ_MODEL = "openai/gpt-oss-120b"
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a key from https://console.groq.com and set it before running."
            )
        _client = Groq(api_key=api_key)
    return _client


INTENT_SYSTEM_PROMPT = """You convert a user's question about student exam results into a single STRICT JSON object. Output ONLY the JSON, nothing else — no markdown fences, no explanation.

Schema:
{
  "intent": one of ["filter", "best", "worst", "subject_marks", "average", "personal_info", "subject_feedback", "compare", "count", "unknown"],
  "field": subject name (e.g. "math"), or "percentage", "total", "roll_no", "gender", "std", or null,
  "operator": one of [">", "<", ">=", "<=", "==", null],
  "value": a number, or null,
  "student_names": array of student name strings mentioned in the question (lowercase), or [],
  "return_field": one of ["name", "marks", "percentage", "roll_no", "gender", "std", "all"]
}

Guidance for turning vague language into the schema:
- "give me student name who got >90%" -> intent=filter, field=percentage, operator=">", value=90, return_field=name
- "good performance student" / "best student" / "topper" -> intent=best, field=percentage
- "worst student" / "weak student" / "who needs help" -> intent=worst, field=percentage
- "how much did X score in math" / "X's math marks" -> intent=subject_marks, field=math, student_names=["x"]
- "average marks in science" -> intent=average, field=science
- "class average" / "overall average" -> intent=average, field=percentage
- "X's roll number" / "what gender is X" / "which std is X in" -> intent=personal_info, field=roll_no|gender|std, student_names=["x"]
- "is X good at math" / "how is X doing in gujarati" / "X's weak subject" -> intent=subject_feedback, field=<subject or null for overall>, student_names=["x"]
- "compare X and Y" -> intent=compare, student_names=["x","y"]
- "how many students scored above 80" -> intent=count, field=percentage, operator=">", value=80

Only use subject names from this known list when the question refers to a subject: __SUBJECTS__
Known class values (std): __STDS__

If the question doesn't map cleanly to the schema, use intent="unknown".
"""


def _extract_json(text):
    text = text.strip()
    # Strip markdown fences if the model added them anyway
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def get_intent(question):
    """Stage 1: question -> intent JSON. Sends ONLY the question text
    plus known subject/std vocabulary — no student records."""
    subjects = db.all_subjects()
    stds = sorted({s["std"] for s in db.list_students() if s["std"]})
    system_prompt = INTENT_SYSTEM_PROMPT.replace(
        "__SUBJECTS__", ", ".join(subjects) or "(none yet)"
    ).replace(
        "__STDS__", ", ".join(stds) or "(none yet)"
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"intent": "unknown"}


def _fmt_names(students):
    return ", ".join(s["name"] for s in students) if students else "No matching students found."


def _subject_pct(student, subject):
    s = student["subjects"].get(subject)
    if not s or not s["max_marks"]:
        return None
    return round((s["marks"] / s["max_marks"]) * 100, 2)


def _resolve_field_value(student, field):
    if field == "percentage":
        return student["percentage"]
    if field == "total":
        return student["total"]
    if field in student["subjects"]:
        return _subject_pct(student, field)
    return None


def execute(intent_obj, question=""):
    """Stage 2 + 3: run the intent against SQLite, then render a
    natural-language sentence from the (already local) result."""
    intent = intent_obj.get("intent", "unknown")
    field = intent_obj.get("field")
    operator = intent_obj.get("operator")
    value = intent_obj.get("value")
    names = intent_obj.get("student_names") or []
    return_field = intent_obj.get("return_field", "name")

    students = db.list_students()
    if not students:
        return "There are no students in the system yet."

    if intent == "filter":
        if field is None or operator is None or value is None:
            return "I couldn't work out the condition in that question — could you rephrase it?"
        matches = []
        for s in students:
            v = _resolve_field_value(s, field)
            if v is None:
                continue
            ok = {
                ">": v > value,
                "<": v < value,
                ">=": v >= value,
                "<=": v <= value,
                "==": v == value,
            }.get(operator, False)
            if ok:
                matches.append(s)
        if not matches:
            return f"No students matched {field} {operator} {value}."
        if return_field == "all":
            return "; ".join(
                f"{s['name']} ({field}: {_resolve_field_value(s, field)})" for s in matches
            )
        return f"Students matching {field} {operator} {value}: {_fmt_names(matches)}"

    if intent == "best":
        f = field or "percentage"
        ranked = sorted(students, key=lambda s: (_resolve_field_value(s, f) or -1), reverse=True)
        top = ranked[0]
        val = _resolve_field_value(top, f)
        return f"{top['name']} has the best {f}, scoring {val}."

    if intent == "worst":
        f = field or "percentage"
        ranked = sorted(students, key=lambda s: (_resolve_field_value(s, f) if _resolve_field_value(s, f) is not None else 999))
        bottom = ranked[0]
        val = _resolve_field_value(bottom, f)
        return f"{bottom['name']} has the lowest {f}, scoring {val}. They may need extra support."

    if intent == "subject_marks":
        if not names:
            return "Which student did you mean?"
        target = db.find_students_by_name(names[0])
        if not target:
            return f"I couldn't find a student named '{names[0]}'."
        s = target[0]
        if field and field in s["subjects"]:
            m = s["subjects"][field]
            return f"{s['name']} scored {m['marks']} out of {m['max_marks']} in {field}."
        return f"{s['name']}'s marks: " + ", ".join(
            f"{subj} {m['marks']}/{m['max_marks']}" for subj, m in s["subjects"].items()
        )

    if intent == "average":
        f = field or "percentage"
        vals = [v for v in (_resolve_field_value(s, f) for s in students) if v is not None]
        if not vals:
            return f"No data available to compute an average for {f}."
        avg = round(sum(vals) / len(vals), 2)
        return f"The average {f} across {len(vals)} student(s) is {avg}."

    if intent == "personal_info":
        if not names:
            return "Which student did you mean?"
        target = db.find_students_by_name(names[0])
        if not target:
            return f"I couldn't find a student named '{names[0]}'."
        s = target[0]
        f = field or "all"
        if f == "roll_no":
            return f"{s['name']}'s roll number is {s['roll_no']}."
        if f == "gender":
            return f"{s['name']}'s gender is recorded as {s['gender']}."
        if f == "std":
            return f"{s['name']} is in std {s['std']}."
        return f"{s['name']}: roll no {s['roll_no']}, std {s['std']}, gender {s['gender']}."

    if intent == "subject_feedback":
        if not names:
            return "Which student did you mean?"
        target = db.find_students_by_name(names[0])
        if not target:
            return f"I couldn't find a student named '{names[0]}'."
        s = target[0]
        if field and field in s["subjects"]:
            pct = _subject_pct(s, field)
            level = db.subject_level(pct)
            phrasing = {
                "good": f"{s['name']} is doing well in {field}, scoring {pct}%.",
                "average": f"{s['name']} has an average grasp of {field}, scoring {pct}%. A bit more practice would help.",
                "weak": f"{s['name']} is struggling with {field}, scoring only {pct}%. This subject needs focused attention.",
            }
            return phrasing[level]
        # No subject given: summarize strongest/weakest subject
        if not s["subjects"]:
            return f"{s['name']} has no marks recorded yet."
        best_subj = max(s["subjects"], key=lambda subj: _subject_pct(s, subj))
        worst_subj = min(s["subjects"], key=lambda subj: _subject_pct(s, subj))
        return (
            f"{s['name']} is strongest in {best_subj} ({_subject_pct(s, best_subj)}%) "
            f"and weakest in {worst_subj} ({_subject_pct(s, worst_subj)}%)."
        )

    if intent == "compare":
        if len(names) < 2:
            return "Please mention two student names to compare."
        a = db.find_students_by_name(names[0])
        b = db.find_students_by_name(names[1])
        if not a or not b:
            return "I couldn't find one or both of those students."
        a, b = a[0], b[0]
        if a["percentage"] == b["percentage"]:
            return f"{a['name']} and {b['name']} are tied at {a['percentage']}%."
        winner, loser = (a, b) if a["percentage"] > b["percentage"] else (b, a)
        return f"{winner['name']} ({winner['percentage']}%) is ahead of {loser['name']} ({loser['percentage']}%)."

    if intent == "count":
        if field is None or operator is None or value is None:
            return "I couldn't work out the condition to count — could you rephrase it?"
        matches = [s for s in students if _resolve_field_value(s, field) is not None and {
            ">": _resolve_field_value(s, field) > value,
            "<": _resolve_field_value(s, field) < value,
            ">=": _resolve_field_value(s, field) >= value,
            "<=": _resolve_field_value(s, field) <= value,
            "==": _resolve_field_value(s, field) == value,
        }.get(operator, False)]
        return f"{len(matches)} student(s) matched {field} {operator} {value}."

    return "I didn't quite understand that question — try asking about a student's marks, percentage, best/worst performer, or a specific subject."


def answer(question):
    intent_obj = get_intent(question)
    return execute(intent_obj, question), intent_obj
