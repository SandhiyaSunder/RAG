# RAG

This repository contains implementations for both the Llama series (LLM) and GPT-Neo (SLM) models,tested using the Retrieval-Augmented Generation (RAG) framework. The project also integrates Streamlit for a user-friendly interface, allowing for interactive experimentation and visualization of the models' performance.


## Features

- **Hugging Face Model Support:**
  - Supports any model available from Hugging Face.
  - Specifically experimented with Llama Series for LLM and GPT-Neo-125M for SLM in this implementation.

- **LlamaIndex Integration:**
  - Utilized for efficient vector database management and retrieval in the RAG pipeline.
  - Enhances the performance of document retrieval and response generation.

- **Streamlit Integration:** 
  - A streamlined user interface for interacting with the models and visualizing results.


## Setup Instructions

To set up and run the RAG implementation, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SandhiyaSunder/RAG.git
   cd RAG
   ```

2. **Install Dependencies:**

Ensure you have pip installed, and then run:
    ```bash 
    pip install -r requirements.txt
    ```

## Run the Models


To run the models, follow these steps:

- **Run the Llama Series Model:**
  - Execute the command:
    ```bash
    streamlit run chat_with_llama.py
    ```
  - This will initialize the Llama series model and open the Streamlit interface.

- **Run the GPT-Neo Model:**
  - Execute the command:
    ```bash
    streamlit run chat_with_gptneo.py
    ```
  - This will initialize the GPT-Neo model and open the Streamlit interface.

## Access the Streamlit Interface
  - After running the above commands, the Streamlit interface will automatically open in your default web browser.

## Using the Streamlit Interface
  - **Provide Input:** Enter your query into the input field.
  - **Receive Response:** The model's response will be displayed after processing your query.
  - **Submit New Query:** Clear the input field and press Enter to submit a new query.


