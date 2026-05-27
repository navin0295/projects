# RBAC Demo

Simple role-based access control (RBAC) Flask API with a Streamlit GUI.

Quickstart

1. Create and activate a Python virtual environment:

   python -m venv .venv
   source .venv/bin/activate

2. Install required packages:

   pip install flask sqlalchemy pyjwt werkzeug streamlit requests

3. Run the API server:

   python rbac.py

4. In a separate terminal, run the Streamlit GUI:

   streamlit run gui.py

Notes

- This repository contains `rbac.py` (Flask API) and `gui.py` (Streamlit frontend).
- I did not modify the code—only added this README. I also scanned the codebase for comment markers like `fix`, `TODO`, or `human written` and found none to remove.

License

This project is provided as-is for demo purposes.
