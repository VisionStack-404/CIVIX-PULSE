import json
from groq import Groq
import os
from dotenv import load_dotenv


load_dotenv(override=True)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"


def agent_filter(user_text):
    print("🧠 AGENT 1 (Groq): Analyzing Conversation...")
    prompt = """
    You are a strict Civic Grievance filtering AI for the government.
    Analyze the text. It is a 'complaint' ONLY if it involves public infrastructure: broken roads, potholes, water leaks, power outages, sanitation/garbage, fallen trees, or public safety hazards.
    If the user talks about a personal problem, a general question, random letters, or says hello, it is NOT a complaint.
    
    - If it IS a civic complaint: Return "is_complaint": true, and "bot_reply": "Got it. Please upload a photo as proof of the issue."
    - If it is NOT a complaint: Return "is_complaint": false, and "bot_reply": "Hello! I am the Civix-Pulse bot. I strictly handle public infrastructure issues like potholes, leaks, or outages. How can I help with your city today?"
    
    Output ONLY strict JSON: {"is_complaint": bool, "bot_reply": "string"}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
            model=GROQ_MODEL, 
            temperature=0.0, # 0.0 makes it strictly logical, no hallucinations
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"❌ Groq A1 Error: {e}")
        return {"is_complaint": False, "bot_reply": "I'm having trouble understanding. Please describe a civic infrastructure issue."}

def agent_triage(complaint_text):
    print("🧠 AGENT 2 (Groq): Triaging Complaint...")
    prompt = "Categorize this civic issue strictly as Water, Electricity, Roads, Sanitation, Public_Safety, or Other. Assign priority_level as CRITICAL, HIGH, or LOW. Output ONLY strict JSON: 'is_legit' (bool), 'category', 'priority_level', 'cluster_tag' (1 word)."
    try:
        res = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": complaint_text}],
            model=GROQ_MODEL, temperature=0.1, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"❌ Groq A2 Error: {e}")
        return {"is_legit": True, "category": "Other", "priority_level": "HIGH", "cluster_tag": "issue"}

def systemic_auditor(complaint_list):
    print("🧠 AGENT 3 (Groq): Running Systemic Audit...")
    prompt = "You are a city infrastructure auditor. Determine the likely root cause of these similar issues in the same area. Output ONLY strict JSON with key: 'root_cause_hypothesis' (1-2 sentences)."
    try:
        res = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Analyze these complaints: {json.dumps(complaint_list)}"}],
            model=GROQ_MODEL, temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"❌ Groq A3 Error: {e}")
        return {"root_cause_hypothesis": "Multiple recurring issues detected in this cluster."}
