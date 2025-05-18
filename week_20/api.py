from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get('/health')
async def health():
    return {
        'status':'healthy',
        'timestamp': datetime.now().isoformat()
    }