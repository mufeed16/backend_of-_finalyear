import logging
from pathlib import Path
from model import load_llm, qa_bot, final_result, stream_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model loads successfully"""
    try:
        logger.info("Testing model loading...")
        llm = load_llm()
        logger.info("✓ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {str(e)}")
        return False

def test_prompt_response():
    """Test if the model can respond to a simple prompt"""
    try:
        logger.info("Testing prompt response...")
        test_query = "What is this document about?"
        response = final_result(test_query)
        
        logger.info("Response received:")
        logger.info("-" * 50)
        logger.info(response)
        logger.info("-" * 50)
        
        logger.info("✓ Prompt test completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Prompt test failed: {str(e)}")
        return False

def test_streaming_response():
    """Test if streaming responses work"""
    try:
        logger.info("Testing streaming response...")
        test_query = "What is this document about?"
        
        logger.info("Streaming response:")
        logger.info("-" * 50)
        for chunk in stream_response(test_query):
            if chunk['type'] == 'token':
                print(chunk['token'], end='', flush=True)
            elif chunk['type'] == 'sources':
                print("\nSources:", chunk['sources'])
        logger.info("\n" + "-" * 50)
        
        logger.info("✓ Streaming test completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Streaming test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running model tests...\n")
    
    # Create necessary directories if they don't exist
    model_path = Path(__file__).parent.parent / "ai-model"
    vector_store_path = Path(__file__).parent / "vectorstoredb_faiss"
    
    model_path.mkdir(exist_ok=True)
    vector_store_path.parent.mkdir(exist_ok=True)
    
    # Run tests
    tests = [
        ("Model Loading", test_model_loading),
        ("Prompt Response", test_prompt_response),
        ("Streaming Response", test_streaming_response)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Print summary
    print("\nTest Summary:")
    print("-" * 50)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    print("-" * 50)