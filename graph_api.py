from fastapi import FastAPI, File, UploadFile
import os
import shutil
import KnowledgeGraph
from vercel import Vercel

app = FastAPI()

# Create temporary folder if it doesn't exist
if not os.path.exists("tmp"):
    os.makedirs("tmp")

@app.get("/")
def home():
    return {"message": "Health Document Graph Generator API"}

@app.post("/generate-graph/")
async def generate_graph(file: UploadFile = File(...)):
    file_location = f"tmp/{file.filename}"

    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"File saved at {file_location}")
    except Exception as e:
        return {"error": f"Error saving file: {e}"}

    try:
        result = KnowledgeGraph.main(file_location)
        print(f"Graph generation result: {result}")
    except Exception as e:
        return {"error": f"Error generating graph: {e}"}

    os.remove(file_location)
    return {"status": "Graph generated successfully", "result": result}

# This wraps the FastAPI app to be used in serverless functions
handler = Vercel(app)
