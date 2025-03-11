from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
from werkzeug.security import generate_password_hash, check_password_hash
from flask_uploads import UploadSet, configure_uploads, ALL
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS
import fitz  # PyMuPDF
from datetime import timedelta, datetime
import os
import json
import magic
from model import final_result
from datacreate import create_vector_db
from config import init_app, Config

from dotenv import load_dotenv
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app = init_app(app)  # Initialize app configuration
CORS(app, origins=["http://localhost:3000"])  # Allow requests from frontend

# Allowed file types
ALLOWED_EXTENSIONS = {'pdf'}
ALLOWED_MIMETYPES = {'application/pdf'}

# Configure file uploads
uploads = UploadSet('uploads', ALL)
configure_uploads(app, uploads)

# Initialize MongoDB
try:
    mongo = PyMongo(app)
    # Test the connection
    mongo.db.command('ping')
    logger.info("MongoDB connection successful")
except Exception as e:
    logger.error(f"MongoDB connection failed: {str(e)}")
    sys.exit(1)

jwt = JWTManager(app)

# Error handlers
@app.errorhandler(500)
def handle_500_error(e):
    return jsonify(error="Internal server error"), 500

@app.errorhandler(404)
def handle_404_error(e):
    return jsonify(error="Resource not found"), 404

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOADS_DEFAULT_DEST']):
    os.makedirs(app.config['UPLOADS_DEFAULT_DEST'])

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')
    role = request.json.get('role', 'user')  # Default to user if not specified
    
    # Check if username already exists
    if mongo.db.users.find_one({'username': username}):
        return jsonify(message="Username already exists"), 400

    hashed_password = generate_password_hash(password)
    mongo.db.users.insert_one({
        'username': username, 
        'password': hashed_password,
        'role': role
    })
    return jsonify(message="User registered successfully"), 201

@app.route('/admin/register', methods=['POST'])
def admin_register():
    username = request.json.get('username')
    password = request.json.get('password')
    admin_key = request.json.get('admin_key')
    
    # Verify admin key (you should change this to a secure key)
    if admin_key != "adminkey":
        return jsonify(message="Invalid admin key"), 403
    
    # Check if username already exists
    if mongo.db.users.find_one({'username': username}):
        return jsonify(message="Username already exists"), 400

    hashed_password = generate_password_hash(password)
    mongo.db.users.insert_one({
        'username': username, 
        'password': hashed_password,
        'role': 'admin'
    })
    return jsonify(message="Admin registered successfully"), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    user = mongo.db.users.find_one({'username': username})

    if user and check_password_hash(user['password'], password):
        access_token = create_access_token(identity=username, additional_claims={'role': user['role']})
        return jsonify(access_token=access_token), 200
    return jsonify(message="Invalid username or password"), 401
@app.route('/documents', methods=['GET'])
# @jwt_required()
def get_documents():
    try:
        documents = list(mongo.db.documents.find({}, {'_id': False}))
        return jsonify(documents)
    except Exception as e:
        return jsonify({'message': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_type(file_stream):
    file_head = file_stream.read(2048)
    file_stream.seek(0)
    mime_type = magic.from_buffer(file_head, mime=True)
    logger.info(f"Detected MIME type: {mime_type}")
    # Fallback: accept application/octet-stream as a valid PDF mime type
    if mime_type == 'application/octet-stream':
        return True
    is_valid_mime_type = mime_type in ALLOWED_MIMETYPES
    logger.info(f"Is valid MIME type: {is_valid_mime_type}")
    return is_valid_mime_type

@app.route('/upload', methods=['POST'])
# @jwt_required()
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
            
        if not file or not allowed_file(file.filename):
            return jsonify({"message": "Invalid file type. Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], filename)

        # Save the file first
        file.save(filepath)

        # Reset file stream position for reading
        file.seek(0)  # This prevents the "read of closed file" error
        
        def generate():
            try:
                logger.info(f"File saved to {filepath}")

                yield "data: " + json.dumps({"step": "upload", "status": "complete", "message": "PDF uploaded successfully"}) + "\n\n"
                
                yield "data: " + json.dumps({"step": "processing", "status": "inProgress", "message": "Processing PDF and extracting text..."}) + "\n\n"
                
                yield "data: " + json.dumps({"step": "embeddings", "status": "inProgress", "message": "Creating text embeddings using sentence transformers..."}) + "\n\n"
                
                # Create vector store
                success = create_vector_db(filepath)
                if not success:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    yield "data: " + json.dumps({"step": "vectordb", "status": "error", "message": "Failed to create vector database"}) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"step": "vectordb", "status": "complete", "message": "Vector database created successfully"}) + "\n\n"
                
            except Exception as e:
                logger.error(f"Error during file processing: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                yield "data: " + json.dumps({"step": "error", "message": str(e)}) + "\n\n"
        
        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/chat-history', methods=['GET'])
# @jwt_required()
def get_chat_history():
    try:
        claims = get_jwt()
        username = get_jwt_identity()
        if claims.get('role') == 'admin':
            # Admins can see all chat history
            chat_history = list(mongo.db.chat_history.find({}, {'_id': 0}).sort('timestamp', -1))
        else:
            # Users can only see their own chat history
            chat_history = list(mongo.db.chat_history.find(
                {'user': username},
                {'_id': 0}
            ).sort('timestamp', -1))
        
        return jsonify(chat_history), 200
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return jsonify(message="Error fetching chat history"), 500
        return jsonify(message="Error fetching chat history"), 500

@app.route('/query', methods=['POST'])
# @jwt_required()
def query():
    try:
        user_query = request.json.get('query')
        
        def generate():
            for chunk in final_result(user_query, stream=True):
                if chunk['type'] == 'token':
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk['token']})}\n\n"
                
            yield "data: {\"type\": \"done\"}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream'
        )
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify(message="Error processing query"), 500

def extract_text_from_pdf(filename):
    text = ""
    with fitz.open(filename) as doc:
        for page in doc:
            text += page.get_text()
    return text

# def create_vector_db(filepath):
#     try:
#         # Open the saved PDF file and process it to create the vector store
#         with open(filepath, 'rb') as pdf_file:
#             # Replace the following with your PDF processing and vector store creation logic
#             pdf_data = pdf_file.read()
#             # For example, process pdf_data here...
#             pass
#         return True
#     except Exception as e:
#         logger.error(f"Error in create_vector_db: {str(e)}")
#         return False

if __name__ == '__main__':
    app.run(debug=True)
