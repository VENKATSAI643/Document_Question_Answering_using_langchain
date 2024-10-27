import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import fitz  # PyMuPDF for PDF text extraction
import aiofiles

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from any origin (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, or specify as ["http://localhost:5500"] if you're testing locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load environment variables
load_dotenv()


# Load Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found. Please check your .env file.")

# Initialize the LangChain model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    tweet_prompt = PromptTemplate.from_template("You are a content creator. Answer the question based on the content: {text}.")
    tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)
except Exception as e:
    raise ValueError(f"Error initializing ChatGoogleGenerativeAI: {e}")

# In-memory storage for extracted PDF text
pdf_text_storage = {}

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text
@app.post("/upload-pdf")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    """Endpoint to upload PDF and extract text."""
    file_name = f"{uuid.uuid4()}.pdf"
    file_path = f"./temp/{file_name}"

    # Ensure temporary directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await pdf_file.read()
            await f.write(content)
            print(f"File successfully saved at {file_path}, size: {len(content)} bytes")
    except Exception as e:
        print(f"Error saving the file: {e}")
        return JSONResponse(content={"error": f"Failed to save PDF file: {e}"}, status_code=500)

    # Extract text from the saved PDF
    try:
        pdf_text = extract_text_from_pdf(file_path)
        print(f"Extracted text from PDF: {pdf_text[:200]}...")  # Display first 200 characters for debugging
        pdf_id = str(uuid.uuid4())
        pdf_text_storage[pdf_id] = pdf_text  # Store the extracted text in memory
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return JSONResponse(content={"error": f"Failed to extract text from PDF: {e}"}, status_code=500)

    # Keep the file on disk instead of deleting it
    # os.remove(file_path)  # Comment out or remove this line if you want to retain the file

    return {"pdf_id": pdf_id, "message": "PDF uploaded and text extracted successfully."}

    # Extract text from the saved PDF

@app.post("/ask-query")
async def ask_query(pdf_id: str = Form(...), query: str = Form(...)):
    """Endpoint to ask a query about a previously uploaded PDF."""
    if pdf_id not in pdf_text_storage:
        return JSONResponse(content={"error": "Invalid PDF ID or PDF not found."}, status_code=404)

    pdf_text = pdf_text_storage[pdf_id]

    # Create a specific prompt template for answering a query
    specific_prompt = PromptTemplate.from_template(
        "Based on the following text, answer the query: {text}. Query: {query}."
    )

    query_chain = LLMChain(llm=llm, prompt=specific_prompt, verbose=True)

    # Run LangChain to answer the query
    try:
        response = query_chain.run(text=pdf_text, query=query)
        print(f"LangChain response: {response}")
    except Exception as e:
        print(f"Failed to process query: {e}")
        return JSONResponse(content={"error": f"Failed to process query: {e}"}, status_code=500)

    return {"response": response}
