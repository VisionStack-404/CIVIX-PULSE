import httpx
import os
from dotenv import load_dotenv

load_dotenv(override=True)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

payload = {
    "assigned_worker": "1a867ec2-e64e-49b0-9fe8-24a8aac2d873"
}

print("Assigning all active open tickets to John Doe in the database...")

try:
    res = httpx.patch(f"{url}/rest/v1/grievances?status=eq.Open", headers=headers, json=payload)
    print(f"Open grievances assignment status: {res.status_code}")
    
    res2 = httpx.patch(f"{url}/rest/v1/grievances?status=eq.Merged", headers=headers, json=payload)
    print(f"Merged grievances assignment status: {res2.status_code}")
    
    print("Done! Checked and updated in Supabase.")
except Exception as e:
    print(f"Error updating database: {e}")
