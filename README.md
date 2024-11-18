# Chat with YouTube Videos

## Overview

**Chat with YouTube Videos** is a web application that allows users to upload a YouTube video link and ask questions about its content. The application downloads the audio from the video, transcribes it, and then uses a language model to answer questions based on the transcription.

## Features

- **YouTube Video Input:** Users can paste a YouTube URL to process.
- **Audio Download:** Automatically downloads the audio from the video.
- **Transcription:** Transcribes the downloaded audio using AssemblyAI.
- **Question Answering:** Uses OpenAI's GPT-4 to answer questions based on the transcription.
- **Chat History:** Keeps track of the conversation history.

## Requirements

- Python 3.8 or higher
- Streamlit
- yt-dlp
- pydub
- openai
- assemblyai
- langchain
- chromadb

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/AABENZ/chat-with-youtube-videos
    cd chat-with-youtube-videos
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**

    Create a `.env` file in the root directory and add your API keys:

    ```plaintext
    ASSEMBLYAI_API_KEY=your_assemblyai_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

2. **Interact with the Application:**

    - Paste a YouTube URL in the input field.
    - Click the "Process Video" button to download the audio, transcribe it, and prepare for questions.
    - Ask questions about the video content in the chat interface.

## Screenshots

![Screenshot 1](https://i.ibb.co/q0vJ14T/1.png)

![Screenshot 2](https://i.ibb.co/86mrwLt/2.png)