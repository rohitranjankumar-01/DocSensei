@echo off
echo [1/4] Creating virtual environment with Python 3.12...
:: Force the use of Python 3.11 specifically
py -3.11 -m venv venv
call venv\Scripts\activate

echo [2/4] Upgrading tools to ensure we get binary wheels...
python -m pip install --upgrade pip setuptools wheel

echo [3/4] Installing dependencies...
pip install -r requirements.txt

echo [4/4] Launching DocSensei...
streamlit run app.py
pause