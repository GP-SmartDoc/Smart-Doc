import pytest
from RAG import RAGEngine
import chromadb


def how_to_use_rag_module():
    """This function is not for usage.
        It shows how to use the rag module
    """
    # 1. create a chromadbclient
    client = chromadb.Client()
    
    # NOTE: creating a client object is very dangerous and should 
    # only be done in testing. Having multiple clients referring to the same storage
    # has undefined behaviour, and does not guarantee data consistency. For this reason 
    # the app should have one instance of client that is injected to any class that needs it.
    # NOTE:  chromadb.Client() creates an in-memory databse that is deleted after program 
    # termination. For production, chromadb.PersistentClient(path) should be used which 
    # stores its data on disk at "path".
    
    # 2. create an instance of rag module and inject the client into it
    rag = RAGEngine(client)
    
    # 3.Add different file types:
    rag.add_txt("example.txt") 
    rag.add_image("example.png")
    rag.add_pdf("example.pdf")
    
    # NOTE rag.add_file() is not implemented.
    # NOTE The add_pdf() function calls the add_image() function under the hood if it contains
    # images. So if the add_pdf() function it could be a problem with add_image()
    
    # 3. retrieve relevant data based on the query (look at description of query)
    answer:dict = rag.query("blabla", 1, 1)
    
    text:list = answer["text"] # A list of retrieved text chunks
    images:list = answer["images"] # A list of retrieved image paths

def how_to_test_with_pytest():
    # 1. create a function for each test case in the form "test_testName", Ex:
    def test_basic_test():
        # 2. make your basic setup
        client = chromadb.Client()
        rag = RAGEngine(client)
        rag.add_pdf("example.pdf")
        
        # 3. compare the expected result with the actual result using assert,
        # if they are not equal, the testcase fails
        answer = rag.query("bla bla", 1, 1)
        assert len(answer["text"]) == 1
    
    # 4. `pip install pytest`, the run the file with `pytest tests.py`. It will show you
    # which tests passed and which didn't. If it succeeds, you'll get ahmed salah congratulation.
def test_pdf_only():
    """Test: Only a PDF is added."""
    client = chromadb.Client()
    rag = RAGEngine(client)
    rag.add_pdf("test3.pdf")

    result = rag.query("What'on the motherboard?", k_text=2, k_image=3)
    print(result)
    assert "text" in result
    assert "images" in result
    assert isinstance(result["text"], list)
    assert len(result["text"]) > 0


def test_two_images_only():
    """Test: Only 2 images are added."""
    client = chromadb.Client()
    rag = RAGEngine(client)
    rag.add_image("Ss1.png")
    rag.add_image("Ss2.png")

    result = rag.query("What are the predicates will be defined?", k_text=4, k_image=1)
    print(result)
    # Images must be retrieved
    assert len(result["images"]) == 2
    assert all(isinstance(path, str) for path in result["images"])

    # No text because no TXT/PDF added
    assert len(result["text"]) == 0


def test_all_types():
    """Test: PDF + images added together."""
    client = chromadb.Client()
    rag = RAGEngine(client)
    rag.add_pdf("test2.pdf")
    rag.add_image("Ss1.png")
    rag.add_image("Ss2.png")

    result = rag.query("What's on the Motherboard?", k_text=3, k_image=1)
    print(result)
    assert len(result["images"]) == 2
    assert len(result["text"]) > 0  # PDF text exists
        
    