import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict
from dotenv import load_dotenv
from hindsight import HindsightClient

# Load environment variables from .env file
load_dotenv()

from templates import BOOKING_AGENT_SYSTEM_PROMPT

# In-memory store to hold the latest agent responses and conversation history
agent_responses: Dict[str, str] = {}
conversation_history: Dict[str, list] = {}

hindsight_client = None

class TranscriptPayload(BaseModel):
    user_id: str
    transcript: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hindsight_client
    print("Connecting to Hindsight Cloud...")
    
    api_key = os.environ.get("HINDSIGHT_API_KEY")
    if not api_key:
        raise RuntimeError("HINDSIGHT_API_KEY must be set in .env")
        
    hindsight_client = HindsightClient(api_key=api_key, base_url="https://api.hindsight.vectorize.io")
    print("Hindsight Cloud connection established.")
    
    yield
    
    print("Shutting down API server...")

app = FastAPI(lifespan=lifespan)

def format_hindsight_memories(results, title="Past Memories") -> str:
    """Formats Hindsight recall results into a readable string for the prompt."""
    if not results or not isinstance(results, list):
        return "No memories found."
        
    lines = [f"--- {title} ---"]
    for r in results:
        # Hindsight results can be objects or dicts depending on the client version
        if hasattr(r, 'content'):
            lines.append(f"• {r.content}")
        elif isinstance(r, str):
            lines.append(f"• {r}")
        elif isinstance(r, dict) and 'content' in r:
            lines.append(f"• {r['content']}")
    lines.append("--------------------------")
    return "\n".join(lines)

@app.post("/receive_transcript")
def receive_transcript(payload: TranscriptPayload):
    try:
        user_id = payload.user_id
        transcript = payload.transcript
        
        print(f"[{user_id}] Received transcript: {transcript}")
        
        # 1. Retrieve context from general knowledge base
        try:
            kb_results = hindsight_client.recall(bank_id="vaani_knowledge_base", query=transcript)
            context_text = format_hindsight_memories(kb_results, "Retrieved Knowledge")
            print(f"[{user_id}] Recalled knowledge from vaani_knowledge_base.")
        except Exception as e:
            print(f"[{user_id}] Warning: Knowledge recall failed: {e}")
            context_text = "No specific itinerary or property info found."
            
        # 2. Retrieve user-specific memory
        try:
            user_results = hindsight_client.recall(bank_id=user_id, query=transcript)
            user_memories = format_hindsight_memories(user_results, "Guest Profile & Memories")
            print(f"[{user_id}] Recalled user memories.")
        except Exception as e:
            print(f"[{user_id}] Warning: User memory recall failed: {e}")
            user_memories = "No guest profile information yet."
        
        system_prompt = BOOKING_AGENT_SYSTEM_PROMPT.format(
            retrieved_context=context_text,
            user_memories=user_memories,
            booking_state="IDLE"
        )
        
        # Maintain conversation history per user
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        # 3. Generate response with Groq
        from groq import Groq
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        client = Groq(api_key=groq_api_key)
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[user_id][-10:]) # Keep last 10 messages
        messages.append({"role": "user", "content": transcript})
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
        )
        agent_output = completion.choices[0].message.content.strip()
        
        # Update conversation history
        conversation_history[user_id].append({"role": "user", "content": transcript})
        conversation_history[user_id].append({"role": "assistant", "content": agent_output})
        
        # 4. Retain new memory to Hindsight Cloud
        try:
            retention_str = f"Guest said: '{transcript}'\nAgent replied: '{agent_output}'"
            print(f"[{user_id}] Retaining interaction to Hindsight Cloud...")
            hindsight_client.retain(bank_id=user_id, content=retention_str)
        except Exception as e:
            print(f"[{user_id}] Warning: Retention failed: {e}")
        
        agent_responses[user_id] = agent_output
        return {"status": "success", "message": "Transcript received and processed via Hindsight Cloud."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_agent_response/{user_id}")
def get_agent_response(user_id: str):
    if user_id not in agent_responses:
        raise HTTPException(status_code=404, detail="No agent response found for this user.")
    
    response_text = agent_responses.pop(user_id)
    
    return {
        "user_id": user_id,
        "response": response_text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
