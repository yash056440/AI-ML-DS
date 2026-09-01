# Result Register — Student results website + NLP chatbot

## What's inside
- `app.py` — Flask app: serves the page and the API
- `db.py` — SQLite data layer (students + marks, percentage/grade always computed live)
- `chatbot.py` — the 3-stage NLP pipeline (question → Groq intent JSON → local query → sentence)
- `templates/index.html` — single-page frontend: **Register** tab (add/edit/delete students) and **Ask** tab (chatbot)
- `students.db` — created automatically on first run

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key from https://console.groq.com, then open **`config.env`** (already included in this folder) in Notepad or any text editor and replace the placeholder:

```
GROQ_API_KEY=your-key-here
```

with your real key, then save the file. No renaming needed.

Run the app:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## How the chatbot stays private and accurate
1. **Question → intent (Groq):** only your typed question, plus the list of subject names and class values you've entered (not any student's personal data), is sent to Groq. It replies with a small JSON object describing what you're asking for (e.g. `{"intent":"filter","field":"percentage","operator":">","value":90}`).
2. **Intent → data (local):** that JSON is used to run a query against your local SQLite database — this step never touches the internet.
3. **Data → sentence (local):** the result is turned into a natural sentence with simple rules (e.g. ≥75% in a subject → "doing well", 40–74% → "average", <40% → "struggling").

## Extending it
- Add more question types by adding a new `intent` value to the schema in `chatbot.py`'s `INTENT_SYSTEM_PROMPT` and a matching branch in `execute()`.
- Change the good/average/weak thresholds in `db.subject_level()`.
- Add authentication before deploying publicly — right now the admin form has no login.
