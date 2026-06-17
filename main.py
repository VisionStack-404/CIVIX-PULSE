from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx
import math
from datetime import datetime, timezone, timedelta
import io
import json
import asyncio
from groq import AsyncGroq
import traceback
import base64


from sentence_transformers import SentenceTransformer
import imagehash
from PIL import Image

import os
from dotenv import load_dotenv


load_dotenv(override=True)



#telegram ,supabase,gqoq tokens#
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
groq_client = AsyncGroq(api_key=GROQ_API_KEY, timeout=25.0)

GROQ_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

user_sessions = {}

print("Loading Local Embedding Model... (This takes a few seconds)")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded! System Ready.")




async def safe_request(method, url, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method == "GET": return await client.get(url, **kwargs)
                elif method == "POST": return await client.post(url, **kwargs)
                elif method == "PATCH": return await client.patch(url, **kwargs)
        except Exception as e:
            if attempt == retries - 1: return None 
            await asyncio.sleep(1.5) 

async def send_message(chat_id, text):
    if not chat_id: return
    if not text: text = "Processing..."
    
    res = await safe_request("POST", f"{TELEGRAM_API_URL}/sendMessage", json={"chat_id": str(chat_id), "text": str(text)})
    if res and res.status_code != 200:
        print(f"❌ Telegram API Error on Chat {chat_id}: {res.text}")
    return res

async def get_telegram_image_bytes(file_id):
    res1 = await safe_request("GET", f"{TELEGRAM_API_URL}/getFile?file_id={file_id}")
    if not res1 or res1.status_code != 200: return None
    file_path = res1.json()['result']['file_path']
    res2 = await safe_request("GET", f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}")
    return res2.content if res2 and res2.status_code == 200 else None



async def transcribe_voice(file_id):
    print("🎤 Transcribing Audio via Groq Whisper...")
    res1 = await safe_request("GET", f"{TELEGRAM_API_URL}/getFile?file_id={file_id}")
    if not res1 or res1.status_code != 200: return None
    file_path = res1.json()['result']['file_path']
    
    res2 = await safe_request("GET", f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}")
    if not res2 or res2.status_code != 200: return None

    try:
        transcription = await groq_client.audio.transcriptions.create(
            file=("audio.ogg", res2.content),
            model="whisper-large-v3-turbo"
        )
        return transcription.text
    except Exception as e:
        print(f"❌ Whisper Error: {e}")
        return None




async def agent_visual_auditor(image_bytes, complaint_text):
    print("👁️ AGENT 4 (Groq Vision): Auditing Image Relevance & Authenticity...")
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB": img = img.convert("RGB")
        img.thumbnail((800, 800)) 
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        optimized_bytes = buf.getvalue()
    except Exception as e:
        optimized_bytes = image_bytes

    base64_image = base64.b64encode(optimized_bytes).decode('utf-8')
    
    prompt = f"""
    You are a strict municipal image verifier. 
    1. Check if the image clearly matches this complaint: "{complaint_text}".
    2. Check if the image looks like a real photo, or if it looks AI-generated/fake.
    3. Extract any readable text (OCR) from signs/buildings in the photo."""
    
    Output strictly JSON:
    {{
        "is_relevant": true/false,
        "is_real": true/false,
        "ocr_text": "any text found or 'None'",
        "reason": "Brief explanation of your verdict"
    }}
    """
    try:
        chat_completion = await asyncio.wait_for(
            groq_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                model=VISION_MODEL, temperature=0.1, response_format={"type": "json_object"}
            ),
            timeout=15.0
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ Vision API Error: {e}")
        return {"is_relevant": True, "is_real": True, "ocr_text": "None", "reason": "Bypassed due to API load."}

# ==========================================
# 🧠 GROQ AI AGENTS (Filters & ADVANCED TRIAGE)

async def agent_filter(user_text):
    print(f"🧠 AGENT 1 (Groq): Filtering incoming text: '{user_text}'")
    prompt = """
    You are a strict Civic Grievance filtering AI.
    Analyze the text. It is a 'complaint' ONLY if it involves public infrastructure: broken roads, potholes, leaks, outages, sanitation, fallen trees, or safety hazards.
    If the user talks about a personal problem, says hello, or asks a generic question, it is NOT a complaint.
    
    Output ONLY strict JSON: {"is_complaint": bool, "bot_reply": "string"}
    - true: "Got it. Please upload a photo as proof of the issue."
    - false: "Hello! I strictly handle public infrastructure issues like potholes, leaks, or outages. Please describe a civic issue."
    """
    try:
        res = await asyncio.wait_for(
            groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt + "\n\nUser text: " + user_text}], model=GROQ_MODEL, temperature=0.0, response_format={"type": "json_object"}),
            timeout=10.0
        )
        result = json.loads(res.choices[0].message.content)
        print(f"🧠 AGENT 1 (Groq) Result: {result}")
        return result
    except Exception as e:
        print(f"❌ Groq Filter Error: {e}")
        return {"is_complaint": False, "bot_reply": "Please describe a civic infrastructure issue."}

async def agent_triage(complaint_text, ocr_text=""):
    print("🧠 AGENT 2 (Groq): Running Multi-Factor Triage Matrix...")
    prompt = f"""
    You are an advanced Civic Triage Engine. Evaluated the following issue: "{complaint_text}". Context from image OCR: "{ocr_text}".

    Calculate a priority score (0-100) based on this Multi-Factor Matrix:
    1. Safety Risk (0-40 points): Imminent danger to life or limb.
    2. Infrastructure Damage (0-30 points): Risk of cascading damage.
    3. Community Impact (0-30 points): Number of people affected.

    Rules for priority_level:
    - Total Score >= 80: CRITICAL
    - Total Score 50-79: HIGH
    - Total Score < 50: LOW

    Strictly choose a category: Water, Electricity, Roads, Sanitation, Public_Safety, Other."""

    Output strictly JSON:
    {{
        "is_legit": true,
        "category": "string",
        "priority_score": int,
        "priority_level": "CRITICAL" | "HIGH" | "LOW",
        "cluster_tag": "1 word summary",
        "reasoning": "1 short sentence justifying the score"
    }}
    """
    try:
        res = await asyncio.wait_for(
            groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=GROQ_MODEL, temperature=0.1, response_format={"type": "json_object"}),
            timeout=10.0
        )
        result = json.loads(res.choices[0].message.content)
        print(f"📊 Matrix Score: {result.get('priority_score')}/100 -> {result.get('priority_level')}")
        return result
    except Exception as e:
        print(f"❌ Groq Triage Error: {e}")
        return {"is_legit": True, "category": "Other", "priority_level": "HIGH", "cluster_tag": "issue"}

# ==========================================
# 🧠 DETERMINISTIC 3-LAYER CLUSTERING ENGINE (ARMORED)
# ==========================================
async def run_clustering_pipeline(lat, lng, complaint_text, file_id, db_headers):
    print("\n🔍 Running 3-Layer Clustering Pipeline (Supabase RPC)...")
    query_embedding = []
    query_hash_str = ""
    query_hash_obj = None

    try:
        # LAYER 1: Text Embedding (Armored)
        try:
            query_embedding = embedding_model.encode(complaint_text).tolist()
        except Exception as emb_e:
            print(f"⚠️ Text Embedding failed: {emb_e}")
            query_embedding = [0.0] * 384 # Mathematical fallback so DB doesn't crash

        # LAYER 2: Image Hashing (Armored)
        try:
            img_bytes = await get_telegram_image_bytes(file_id)
            if img_bytes:
                query_hash_obj = imagehash.phash(Image.open(io.BytesIO(img_bytes)))
                query_hash_str = str(query_hash_obj)
        except Exception as img_e:
            print(f"⚠️ Image Hashing skipped (bad file type): {img_e}")

        # LAYER 3: Database Search
        payload = {"query_lat": lat, "query_lng": lng, "query_embedding": query_embedding}
        res = await safe_request("POST", f"{SUPABASE_URL}/rest/v1/rpc/find_nearby_similar_tickets", headers=db_headers, json=payload)
        
        if not res or res.status_code != 200:
            print(f"🚨 Supabase RPC Failed! {res.text if res else 'Timeout'}")
            return None, query_embedding, query_hash_str

        potential_clusters = res.json()
        if not potential_clusters: 
            print("✅ Layer 1 & 2 Result: Database found no nearby tickets. Creating NEW ticket.")
            return None, query_embedding, query_hash_str

        print(f"⚠️ Layer 1 & 2 Passed! Found {len(potential_clusters)} nearby tickets. Entering Image Check...")

        # LAYER 3: Image Comparison
        for t in potential_clusters:
            db_hash_str = t.get('image_hash')
            if not db_hash_str or not query_hash_obj:
                print(f"   -> Skipping Ticket {t.get('id', 'Unknown').split('-')[0]}: Missing hash data.")
                continue
                
            db_hash = imagehash.hex_to_hash(db_hash_str)
            hash_diff = query_hash_obj - db_hash
            
            print(f"   -> DB Ticket {t['id'].split('-')[0]}: Dist={t.get('distance_km', 0):.3f}km | TextMatch={t.get('similarity', 0)*100:.1f}% | ImageDiff={hash_diff}")

            if hash_diff <= 35: 
                print(f"🚨 DUPLICATE CONFIRMED! Image Difference ({hash_diff}) is under threshold.")
                return t, query_embedding, query_hash_str 

        print("✅ Layer 3 Result: Images were too different. Creating NEW ticket.")
        return None, query_embedding, query_hash_str

    except Exception as e:
        print(f"❌ FATAL Clustering System Error: {e}")
        traceback.print_exc()
        if not query_embedding: query_embedding = [0.0] * 384
        return None, query_embedding, query_hash_str

def calculate_remaining_eta(created_at_iso, priority):
    created_time = datetime.fromisoformat(created_at_iso.replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    target_time = created_time + timedelta(hours=2 if priority == "CRITICAL" else 24 if priority == "HIGH" else 48)
    remaining = target_time - now
    if remaining.total_seconds() <= 0: return "Overdue - Worker should be on site."
    hours, remainder = divmod(remaining.seconds, 3600)
    return f"{remaining.days}d {hours}h" if remaining.days > 0 else f"{hours}h {remainder//60}m"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

async def get_nearest_zone(lat, lng, db_headers):
    res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/locations", headers=db_headers)
    if not res or res.status_code != 200: return None, "Unknown"
    locations_list = res.json()
    if not locations_list:
        print("⚠️ Warning: No locations found in the 'locations' table! Please seed some locations in Supabase.")
        return None, "Unknown"
    closest_loc, min_dist = None, float('inf')
    for loc in locations_list:
        dist = haversine(lat, lng, loc['center_lat'], loc['center_lng'])
        if dist < min_dist: min_dist, closest_loc = dist, loc
    if not closest_loc: return None, "Unknown"
    return closest_loc['id'], closest_loc['name']

# ==========================================
# 🚀 MASS RESOLUTION & BROADCAST
# ==========================================
class ResolutionData(BaseModel):
    task_id: str 
    resolution_image_url: str
    worker_id: str

@app.post("/agent-workflow/verify-resolution")
async def trigger_agent_3(data: ResolutionData):
    db_headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    
    t_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{data.task_id}", headers=db_headers)
    if not t_res or not t_res.json(): return {"status": "error"}
    target_ticket = t_res.json()[0]
    
    tickets_to_resolve = [target_ticket]
    cluster_id_to_search = target_ticket.get('cluster_id') or data.task_id

    c_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?cluster_id=eq.{cluster_id_to_search}&status=eq.Merged", headers=db_headers)
    if c_res and c_res.status_code == 200 and c_res.json():
        for child in c_res.json():
            if child['id'] != target_ticket['id']: tickets_to_resolve.append(child)

    for t in tickets_to_resolve:
        cit_id = t['citizen_chat_id']
        caption = "✅ *Civix-Pulse Update*\nThe grievance in your area is officially RESOLVED! Here is the fix:"
        print(f"[BROADCAST] Sending resolution proof to Citizen ID: {cit_id}")
        
        photo_res = await safe_request("POST", f"{TELEGRAM_API_URL}/sendPhoto", json={"chat_id": str(cit_id), "photo": data.resolution_image_url, "caption": caption})
        
        if not photo_res or photo_res.status_code != 200:
            fallback_msg = f"✅ *Civix-Pulse Update*\nYour reported issue is RESOLVED! \nView the repair photo here: {data.resolution_image_url}"
            await send_message(cit_id, fallback_msg)

        await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{t['id']}", headers=db_headers, json={"status": "Resolved", "resolution_image_url": data.resolution_image_url})

    worker_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/workers?id=eq.{data.worker_id}&select=*", headers=db_headers)
    worker_data = worker_res.json()[0] if worker_res and worker_res.status_code == 200 and worker_res.json() else None

    if worker_data:
        cat, loc_id = worker_data['skill_category'], worker_data['location_id']
        queue_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?status=eq.Open&assigned_worker=is.null&category=eq.{cat}&location_id=eq.{loc_id}&order=created_at.asc&limit=1", headers=db_headers)
        pending_tasks = queue_res.json() if queue_res and queue_res.status_code == 200 else []

        if pending_tasks:
            next_task = pending_tasks[0]
            await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{next_task['id']}", headers=db_headers, json={"assigned_worker": data.worker_id})
            
            print(f"[DISPATCH] Notifying Citizen {next_task['citizen_chat_id']} of incoming queued worker...")
            await send_message(next_task['citizen_chat_id'], f"👷 *DISPATCH UPDATE*\nA unit has finished their previous job and is now en route to resolve your ticket!")
            
            worker_telegram = worker_data.get('telegram_chat_id') or worker_data.get('chat_id')
            if worker_telegram:
                print(f"[DISPATCH] Queuing next job for Worker {worker_telegram}...")
                await send_message(worker_telegram, f"🚨 QUEUED DISPATCH 🚨\nPriority: {next_task['priority_level']}\nCheck your portal for details.")
        else:
            await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/workers?id=eq.{data.worker_id}", headers=db_headers, json={"status": "Available"})

    return {"status": "success"}

# ==========================================
# 🛡️ THE BULLETPROOF TELEGRAM HANDLER
# ==========================================
async def process_telegram_update(message):
    try:
        chat_id = message["chat"]["id"]
        raw_text = message.get("text") or message.get("caption") or ""
        user_text = str(raw_text).lower().strip()
        print(f"💬 Processing message from Chat ID {chat_id}: '{user_text}'")
        
        if user_text in ["/start", "cancel", "reset", "restart"]:
            user_sessions[chat_id] = {"step": "waiting_for_text", "complaint_text": "", "photo_id": None, "ocr_text": ""}
            await send_message(chat_id, "Welcome to Civix-Pulse! 🏙️\nPlease describe the civic issue you are facing (or send a Voice Note).")
            return

        if chat_id not in user_sessions: 
            user_sessions[chat_id] = {"step": "waiting_for_text", "complaint_text": "", "photo_id": None, "ocr_text": ""}
        session = user_sessions[chat_id]

        if session["step"] == "waiting_for_text":
            if "voice" in message:
                await send_message(chat_id, "🎙️ Transcribing voice note...")
                transcribed_text = await transcribe_voice(message["voice"]["file_id"])
                if transcribed_text:
                    user_text = transcribed_text
                    await send_message(chat_id, f"📝 I heard: '{transcribed_text}'")
                else:
                    await send_message(chat_id, "Sorry, I couldn't understand the audio. Please type it out.")
                    return

            if user_text:
                analysis = await agent_filter(user_text)
                if analysis.get("is_complaint"):
                    session["complaint_text"] = user_text
                    session["step"] = "waiting_for_photo"
                await send_message(chat_id, analysis.get("bot_reply", "Hi!"))
            else: 
                await send_message(chat_id, "Please type or speak a description of the issue first.")
                
        elif session["step"] == "waiting_for_photo":
            photo_id = None
            if "photo" in message and isinstance(message["photo"], list) and len(message["photo"]) > 0:
                photo_id = message["photo"][-1].get("file_id")
            else:
                for media_type in ["document", "sticker", "video", "animation"]:
                    if media_type in message and isinstance(message[media_type], dict):
                        photo_id = message[media_type].get("file_id")
                        break

            if photo_id:
                await send_message(chat_id, "👁️ Analyzing photo for relevance & authenticity...")
                
                img_bytes = await get_telegram_image_bytes(photo_id)
                if not img_bytes:
                    await send_message(chat_id, "❌ Error downloading image from Telegram. Please try again.")
                    return

                vision_analysis = await agent_visual_auditor(img_bytes, session['complaint_text'])
                
                if not vision_analysis.get("is_relevant") or not vision_analysis.get("is_real"):
                    await send_message(chat_id, f"❌ Photo Rejected.\nReason: {vision_analysis.get('reason')}\n\nPlease upload a real, clear photo of the exact issue.")
                    return
                
                session["photo_id"] = photo_id
                session["ocr_text"] = vision_analysis.get("ocr_text", "")
                session["step"] = "waiting_for_location"
                await send_message(chat_id, "Photo verified! ✅\n📍 Finally, send your Live Location Pin.")
            else:
                if user_text:
                    await send_message(chat_id, "I see you sent text, but I am waiting for visual proof! Please upload a photo/image using the attachment icon.")
                else:
                    keys_found = ", ".join(message.keys())
                    await send_message(chat_id, f"⚠️ I couldn't detect a photo payload. Telegram sent this data type: [{keys_found}].\n\nPlease explicitly attach a photo using the paperclip icon.")

        elif session["step"] == "waiting_for_location":
            if "location" in message:
                lat, lng = message["location"]["latitude"], message["location"]["longitude"]
                await send_message(chat_id, "Initiating 3-Layer Clustering & Analysis... ⏳")
                
                db_headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
                loc_id, loc_name = await get_nearest_zone(lat, lng, db_headers)
                
                duplicate_ticket, vec_emb, img_hash = await run_clustering_pipeline(lat, lng, session['complaint_text'], session['photo_id'], db_headers)
                
                if duplicate_ticket:
                    parent_id = duplicate_ticket['id']
                    p_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/grievances?id=eq.{parent_id}", headers=db_headers)
                    parent_data = p_res.json()[0] if p_res and p_res.json() else None
                    
                    if parent_data:
                        cluster_id = parent_data.get('cluster_id') or parent_id
                        ticket_data = {
                            "citizen_chat_id": str(chat_id), "original_complaint": session['complaint_text'],
                            "category": parent_data['category'], "priority_level": parent_data['priority_level'], 
                            "cluster_id": cluster_id, "extracted_location": loc_name, "location_id": loc_id, 
                            "status": "Merged", 
                            "lat": lat, "lng": lng, "citizen_image_url": session['photo_id'], 
                            "assigned_worker": parent_data['assigned_worker'],
                            "text_embedding": vec_emb, "image_hash": img_hash
                        }
                        
                        db_res = await safe_request("POST", f"{SUPABASE_URL}/rest/v1/grievances", headers=db_headers, json=ticket_data)
                        if not db_res or db_res.status_code not in [200, 201]:
                            print(f"❌ FATAL DB Insert Error (Merged): {db_res.text if db_res else 'Timeout'}")

                    worker_name = duplicate_ticket.get('workers', {}).get('name', 'an active unit')
                    eta_remaining = calculate_remaining_eta(parent_data['created_at'], parent_data['priority_level']) if parent_data else "24 Hours"
                    
                    print(f"[MERGE] Notifying Citizen {chat_id} of duplicate ticket resolution ETA.")
                    msg = f"🚨 *Existing Issue Detected!*\n\nOur mapping confirms this issue is already in our system and grouped with a verified cluster.\n👷 Assigned Unit: {worker_name}\n⏱️ Remaining ETA: {eta_remaining}\n\nYou will receive a photo update the moment it is resolved!"
                    await send_message(chat_id, msg)
                    del user_sessions[chat_id]
                    return 

                ai_result = await agent_triage(session['complaint_text'], session['ocr_text'])
                if ai_result and ai_result.get("is_legit"):
                    category, priority = ai_result.get('category', 'Other'), ai_result.get('priority_level', 'HIGH')
                    eta = "48 Hours" if priority == "LOW" else "24 Hours" if priority == "HIGH" else "2 Hours"
                    
                    assigned_worker_id, worker_name = None, "Pending Assignment"
                    
                    # Dynamically dispatch to the nearest available worker matching the category and zone
                    w_res = await safe_request("GET", f"{SUPABASE_URL}/rest/v1/workers?skill_category=eq.{category}&status=eq.Available&location_id=eq.{loc_id}&limit=1", headers=db_headers)
                    if w_res and w_res.status_code == 200 and len(w_res.json()) > 0:
                        worker = w_res.json()[0]
                        assigned_worker_id = worker.get('id')
                        worker_name = worker.get('name', 'Unit Dispatched')
                        worker_telegram = worker.get('telegram_chat_id') or worker.get('chat_id')
                        
                        await safe_request("PATCH", f"{SUPABASE_URL}/rest/v1/workers?id=eq.{assigned_worker_id}", headers=db_headers, json={"status": "Dispatched"})
                        
                        if worker_telegram:
                            print(f"[DISPATCH] Sending dispatch orders to Worker ID: {worker_telegram}")
                            await send_message(worker_telegram, f"🚨 NEW DISPATCH 🚨\nZone: {loc_name}\nCategory: {category}\nPriority: {priority}\nETA: {eta}")

                    ticket_data = {
                        "citizen_chat_id": str(chat_id), "original_complaint": session['complaint_text'],
                        "category": category, "priority_level": priority, "cluster_id": ai_result.get('cluster_tag'),
                        "extracted_location": loc_name, "location_id": loc_id, "status": "Open",
                        "lat": lat, "lng": lng, "citizen_image_url": session['photo_id'], "assigned_worker": assigned_worker_id,
                        "text_embedding": vec_emb, "image_hash": img_hash
                    }
                    
                    # 🔥 The Final Safety Check: Did Supabase actually save it?
                    db_res = await safe_request("POST", f"{SUPABASE_URL}/rest/v1/grievances", headers=db_headers, json=ticket_data)
                    if db_res and db_res.status_code in [200, 201]:
                        print(f"[DISPATCH] Notifying Citizen {chat_id} of created ticket.")
                        await send_message(chat_id, f"✅ Official Ticket Created!\n📍 Zone: {loc_name}\n📋 Category: {category}\n🚨 Matrix Priority: {priority}\n👷 Unit: {worker_name}\n⏱️ Target ETA: {eta}")
                    else:
                        print(f"❌ FATAL DB Insert Error (New): {db_res.text if db_res else 'Timeout'}")
                        await send_message(chat_id, "⚠️ AI processed the issue, but the database rejected the ticket creation. Please try again.")

                else: 
                    await send_message(chat_id, "❌ AI determined this is not a valid civic complaint.")
                del user_sessions[chat_id]
            else: 
                await send_message(chat_id, "Please share an actual Location Pin using the attachment menu.")

    except Exception as e:
        print(f"\n[FATAL BUG CAUGHT] The background task crashed: {e}")
        traceback.print_exc()
        try:
            chat_id = message["chat"]["id"]
            user_sessions[chat_id] = {"step": "waiting_for_text", "complaint_text": "", "photo_id": None, "ocr_text": ""}
            await send_message(chat_id, "⚠️ The system experienced a brief glitch. Your session has been reset. Please type /start to try again.")
        except: pass

@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print(f"📥 Received Webhook Payload: {json.dumps(data)}")
    if "message" in data: background_tasks.add_task(process_telegram_update, data["message"])
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
