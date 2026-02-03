# AI-Powered Intelligence Digest System

A local, privacy-first AI intelligence platform that aggregates content from Hacker News and RSS feeds, evaluates it using a local LLM (Llama 3.1), and delivers a personalized digest via Email And Telegram.

## Prerequisites

1. **Python 3.11+** installed.
2. **Ollama** installed and running. [Download Ollama](https://ollama.com/download).

## Installation

1. Open a terminal in this directory (`c:\Users\BandiPreethamReddy\Desktop\AI-DEG`).
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Pull the necessary LLM model:
   ```powershell
   ollama pull llama3.1
   ```
   *Note: This downloads ~4.7GB. Ensure you have space.*

## Configuration

1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Edit `.env` with your settings.

### Email Setup (Gmail Example)

To enable email delivery, you need an **App Password** (not your regular login password).

1. Go to your **Google Account** settings.
2. Search for **"App Passwords"** (or go to Security > 2-Step Verification > App passwords).
   *Note: You must have 2-Step Verification enabled.*
3. Create a new app password:
   - **App**: Mail
   - **Device**: Other (Name it "AI Digest")
4. Copy the 16-character password generated.
5. In your `.env` file:
   ```ini
   EMAIL_ENABLED=true
   EMAIL_FROM=your.email@gmail.com
   EMAIL_TO=your.email@gmail.com
   EMAIL_PASSWORD=xxxx xxxx xxxx xxxx  <-- Paste the code here
   ```

## Running the Application

### Option 1: One-Click Start (Recommended)
Double-click the **`start.bat`** file in the project folder. 
It will open two windows (Backend and Frontend) and the app will be ready at `http://localhost:5173`.

### Option 2: Manual Start
You need two separate terminal windows.

**Terminal 1 (Backend):**
```powershell
python -m src.api
```

**Terminal 2 (Frontend):**
```powershell
cd ui
npm run dev
```

---

**Old CLI Method:**
To run the CLI pipeline once without the web UI:
   ```powershell
   python -m src.main
   ```

## Troubleshooting

- **Ollama Connection Error**: Ensure Ollama is running on `localhost:11434`.
- **Email Failed**: Check `app.log` in `data/` folder. Verify your App Password.
