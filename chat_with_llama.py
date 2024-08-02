from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
import streamlit as st
import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext
import os
import tempfile
import time
from transformers import LlamaTokenizer, AutoModelForCausalLM,AutoTokenizer
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens

def count_tokens(text):
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return len(token_ids[0])

# Function to create and return the LLM object
@st.cache_resource
def create_llm():
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    
    # Define LLM with configuration
    llm = HuggingFaceLLM(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        query_wrapper_prompt=PromptTemplate("\n\n\n{query_str}\n\n"),
        context_window=3900,
        max_new_tokens=256,
        # model_kwargs={"quantization_config": quantization_config},
        model_kwargs={"torch_dtype": torch.bfloat16},
        
        generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
        device_map="auto",
    )
    
    return llm

@st.cache_resource
def initialize_service_context(_llm):
    return ServiceContext.from_defaults(
        llm=_llm,
        embed_model="local:BAAI/bge-small-en-v1.5"
    )

@st.cache_resource
def create_vector_index(_service_context, pdf_files):
    with tempfile.TemporaryDirectory() as tmp_dir:
        documents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(tmp_dir, pdf_file.name)
            with open(file_path, 'wb') as f:
                f.write(pdf_file.getbuffer())
            
            loaded_documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            for doc in loaded_documents:
                doc.metadata = {'source': pdf_file.name}
            documents.extend(loaded_documents)
    
    return VectorStoreIndex.from_documents(documents, service_context=_service_context)

# Streamlit UI setup
st.title("Chat with PDFs")
st.caption("This app allows you to chat with PDFs using Llama running locally!")
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Initialize LLM
llm = create_llm()
service_context = initialize_service_context(llm)

if pdf_files:
    st.success(f"Added {len(pdf_files)} files to knowledge base!")

    vector_index = create_vector_index(service_context, pdf_files)
    print("vec: ", vector_index)
    
    # query_engine = vector_index.as_query_engine(response_mode="refine")
    query_engine = vector_index.as_query_engine(response_mode="compact_accumulate")
    print("query: ", query_engine)

    if 'counter' not in st.session_state:
        st.session_state.counter = 1
    if 'history' not in st.session_state:
        st.session_state.history = []

    unique_key = f"query_{st.session_state.counter}"
    prompt = st.text_input(f"Enter your query {st.session_state.counter} (or type 'exit' to quit): ", key=unique_key)
        
    if prompt.lower() == 'exit':
        st.stop()

    if prompt:
        start_time = time.time()  
        response = query_engine.query(prompt)
        end_time = time.time()  

        full_response = response.response
        start_index = full_response.find("Response 1:") + len("Response 1:")
        response_content = full_response[start_index:].strip()

        source_docs = response.source_nodes
        source_names = [doc.node.metadata['source'] for doc in source_docs]
        relevance_scores = [(doc.score, doc.node.metadata['source']) for doc in source_docs]
        relevance_scores.sort(reverse=True, key=lambda x: x[0])

        top_score, top_source = relevance_scores[0]
        duration = end_time - start_time
        tokens = tokenize_text(full_response)
        tokencount = count_tokens(full_response)

        print(f"Number of tokens: {tokencount}")
        print("duration: ", duration)
        
        tokens_per_second = tokencount / duration
        st.session_state.history.append({
            'counter': st.session_state.counter,
            'query': prompt,
            'response': response_content,
            'source': top_source,
            'score': top_score,
            'duration': duration,
            'tokens_per_second' : tokens_per_second
        })
        st.write(f"**Query {st.session_state.counter}:** {prompt}")
        st.write(f"**Response {st.session_state.counter}:** {response_content}", unsafe_allow_html=True)

        st.write(f"**Source Document:** {top_source}")
        st.write(f"**Score:** {top_score}")

        st.session_state.counter += 1

        # Calculate tokens per second
       
        st.write(f"**Duration:** {duration:.2f}")
        st.write(f"**Tokens per Second:** {tokens_per_second:.2f}")

if st.session_state.counter > 1:
    st.write("## Query History")
    for entry in st.session_state.history:
        st.write(f"**Query {entry['counter']}:** {entry['query']}")
        st.write(f"**Response {entry['counter']:} {entry['response']}", unsafe_allow_html=True)
        st.write(f"**Source Document:** {entry['source']}")
        st.write(f"**Score:** {entry['score']}")
        st.write(f"**Duration:** {entry['duration']}")
        st.write(f"**Tokens per second:** {entry['tokens_per_second']}")

