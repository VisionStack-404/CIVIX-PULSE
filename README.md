# 🏙️ Civix-Pulse: AI-Driven Grievance Triage & Dispatch

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white" alt="Supabase" />
  <img src="https://img.shields.io/badge/AI_Inference-Groq-f55036?style=for-the-badge" alt="Groq" />
</div>

> **Zero-Touch Civic Maintenance:** Replacing manual ticketing with instant, AI-driven triage, deterministic clustering, and automated dispatch. Built for SDG 9 (Infrastructure) and SDG 11 (Sustainable Cities).

Civix-Pulse is an enterprise-grade, multi-agent AI system designed to modernize public infrastructure reporting. It allows citizens to report issues via Telegram (Text, Photo, or Voice), uses Vision AI to verify the claim, clusters duplicate reports mathematically, and dispatches city workers based on an automated Risk Matrix.

---

## 📖 Table of Contents
1. [Core Features](#-core-features)
2. [Architecture & Tech Stack](#-architecture--tech-stack)
3. [Prerequisites & Getting API Keys](#-prerequisites--getting-api-keys)
4. [Step 1: Database Setup (Supabase)](#-step-1-database-setup-supabase)
5. [Step 2: Installation & Setting Up Keys](#-step-2-installation--setting-up-keys)
6. [Step 3: Running the System (The 3 Terminals)](#-step-3-running-the-system-the-3-terminals)
7. [How to Use the System](#-how-to-use-the-system)

---

## 🚀 Core Features

* **Omnichannel Ingestion:** Citizens can report issues via text, photos, or voice notes (transcribed instantly via **Groq Whisper-Large-V3**).
* **Agent 4 (Visual Auditor):** Powered by **Meta Llama 4 Scout Vision**, every uploaded photo undergoes OCR and a relevance check to filter out spam and deepfakes before hitting the database.
* **Deterministic 3-Layer Clustering:** We eliminate duplicate truck dispatches using mathematical proof, not LLM guessing:
  1. **Geospatial (Haversine):** 300m radius check.
  2. **Semantic (all-MiniLM-L6):** Cosine similarity on complaint text embeddings.
  3. **Visual (pHash):** Perceptual image hashing to mathematically match photos from different angles.
* **Multi-Factor Triage Matrix:** Replaces subjective human dispatchers with a deterministic scoring engine (Safety Risk, Infrastructure Damage, Community Impact) to auto-flag `CRITICAL` emergencies.

---

## 🏗️ Architecture & Tech Stack

* **Backend Engine:** Python, FastAPI (Fully Asynchronous Event Loop)
* **Frontend Portal:** React.js, Vite, Tailwind CSS
* **Database:** Supabase (PostgreSQL with `pgvector` for semantic search)
* **AI Agents:** Groq LPU (Llama 3.3 70B, Meta Llama 4 Scout 17B)
* **Clustering Libraries:** `sentence-transformers`, `imagehash`, `Pillow`

---

## 🔑 Prerequisites & Getting API Keys

Before writing any code, you must generate the free API keys that power this system.

### 1. Telegram Bot Token
We need a bot to talk to the citizens.
* Open the Telegram app and search for `@BotFather` (verified with a blue checkmark).
* Send the message `/newbot` and follow the prompts to name your bot.
* BotFather will reply with an **HTTP API Token** (e.g., `123456789:ABCdef...`). Save this.

### 2. Supabase URL & Key
This handles our PostgreSQL database and Vector math.
* Go to [Supabase.com](https://supabase.com/) and create a free account and a new project.
* In your project dashboard, click the **Settings** (gear icon) -> **API**.
* Copy the **Project URL** and the **`anon` `public` API Key**. Save these.

### 3. Groq API Key
This powers our ultra-fast Llama 3 & Vision models.
* Go to the [Groq Console](https://console.groq.com/) and sign up for a free account.
* Navigate to **API Keys** on the left menu and click "Create API Key".
* Copy the generated key. Save this.

### 4. Ngrok Tunneling
This exposes your local code to the internet so Telegram can reach it.
* Go to [Ngrok.com](https://ngrok.com/) and sign up.
* Download Ngrok for your OS and follow their 1-step dashboard instruction to authenticate your terminal (`ngrok config add-authtoken <YOUR_TOKEN>`).

---

## 🗄️ Step 1: Database Setup (Supabase)

1. Log into Supabase and open your project.
2. Go to the **SQL Editor** on the left menu.
3. Run this command to enable vector math: `CREATE EXTENSION vector;`
4. Create your `grievances`, `workers`, and `locations` tables according to your schema.
5. **CRITICAL:** Run this custom RPC function to enable the 3-Layer Clustering Engine inside the database:

<details>
<summary><b>Click to view the SQL Clustering Function</b></summary>

```sql
CREATE OR REPLACE FUNCTION find_nearby_similar_tickets(
  query_lat float, query_lng float, query_embedding vector(384)
) RETURNS TABLE (
  id uuid, original_complaint text, image_hash text, created_at timestamptz,
  priority_level text, assigned_worker uuid, workers json,
  distance_km float, similarity float
) LANGUAGE sql STABLE AS $$
  SELECT 
    g.id, g.original_complaint, g.image_hash, g.created_at, 
    g.priority_level, g.assigned_worker, json_build_object('name', w.name) as workers,
    ( 6371 * acos( least(1.0, cos( radians(query_lat) ) * cos( radians( g.lat ) ) * cos( radians( g.lng ) - radians(query_lng) ) + sin( radians(query_lat) ) * sin( radians( g.lat ) ) ) ) ) as distance_km,
    1 - (g.text_embedding <=> query_embedding) as similarity
  FROM grievances g
  LEFT JOIN workers w ON g.assigned_worker = w.id
  WHERE g.status IN ('Open', 'Merged')
    AND g.text_embedding IS NOT NULL
    AND ( 6371 * acos( least(1.0, cos( radians(query_lat) ) * cos( radians( g.lat ) ) * cos( radians( g.lng ) - radians(query_lng) ) + sin( radians(query_lat) ) * sin( radians( g.lat ) ) ) ) ) <= 0.3
    AND 1 - (g.text_embedding <=> query_embedding) > 0.60
  ORDER BY similarity DESC;
$$;
```
</details>

---

## 💻 Step 2: Installation & Setting Up Keys

First, clone the repository to your local machine:
```bash
git clone https://github.com/Visionstack-404/CIVIX-PULSE.git
cd CIVIX-PULSE
```

### 🛑 CRITICAL: Add Your API Keys Here
You must place the keys you generated earlier into a hidden environment file. **Never push this file to GitHub.**

1. In the root folder of the project (right next to `main.py`), create a new text file named exactly `.env`.
2. Open the `.env` file and paste your keys inside exactly like this (no quotes):

```env
TELEGRAM_TOKEN=your_botfather_token_here
SUPABASE_URL=[https://your-project-url.supabase.co](https://your-project-url.supabase.co)
SUPABASE_KEY=your_supabase_anon_public_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃 Step 3: Running the System (The 3 Terminals)

To run the full stack locally, you need to open **three separate terminal windows**.

### 🟢 Terminal 1: The AI Backend
This runs the FastAPI Python server that handles the AI logic.
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (Windows)
venv\Scripts\activate
# (On Mac/Linux, use: source venv/bin/activate)

# 3. Install Python dependencies
pip install fastapi uvicorn httpx sentence_transformers imagehash Pillow groq pydantic python-dotenv

# 4. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```
*(Leave this terminal running!)*

### 🔵 Terminal 2: The Ngrok Webhook Tunnel
Telegram is on the internet, but your Python server is stuck on your local laptop. Ngrok creates a secure bridge to connect them.
1. Open a new terminal.
2. Run this command:
```bash
ngrok http 8000
```
3. Ngrok will give you a "Forwarding" URL that looks like `https://random-words.ngrok-free.app`. **Copy this URL.**
4. Open your web browser (Chrome/Edge) and paste this link, replacing the brackets with your actual Telegram Token and Ngrok URL:
```text
[https://api.telegram.org/bot](https://api.telegram.org/bot)<YOUR_TELEGRAM_TOKEN>/setWebhook?url=<YOUR_NGROK_URL>/webhook
```
Hit Enter. If the browser says `{"ok":true,"description":"Webhook was set"}`, Telegram is successfully connected to your Python code!

### 🟡 Terminal 3: The React Frontend
This runs the web portal for city admins and workers.
1. Open a third terminal and navigate to the frontend folder:
```bash
cd frontend
```
2. Install the Node packages and start the Vite development server:
```bash
npm install
npm run dev
```
3. Click the local host link (usually `http://localhost:5173`) to view the frontend dashboards.

---

## 📱 How to Use the System

1. **Citizen Reporting:** Open Telegram on your phone and search for the bot you created. Type `/start`.
2. **Ingestion:** Send a text description or a voice note of a civic issue (e.g., "There is a massive pothole on Main Street").
3. **Verification:** Upload an image of the issue using the paperclip icon. Agent 4 will audit it.
4. **Location:** Send your Live Location pin via Telegram's attachment menu.
5. **Resolution:** The AI Triage Engine will calculate the priority, mathematically check for duplicates, and auto-dispatch the nearest available city worker. Workers can view tasks on the React portal and resolve them, triggering an automatic photographic broadcast back to the citizens.

---
### 👨‍💻 Built By
**Visionstack-404** 📧 varunvadlakonda4@gmail.com,
**ScriptOrbit-132** 📧 srivarshiniyadav30@gmail.com

*Built with ❤️ for a better, automated future. (SDG 9 & SDG 11)*
