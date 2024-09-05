# YouTube_Video_Analyzer

## Overview
The YouTube Video Summarizer project is designed to help users quickly grasp the content of YouTube videos by providing a summary, key phrases, and sentiment analysis from the video transcript. This tool extracts the video transcript, generates a summary, extracts key phrases, and performs sentiment analysis to visualize the emotional tone throughout the video.

## Features
- **Transcript Retrieval**: Fetches the transcript of a YouTube video using the YouTube Transcript API.
- **Summarization**: Generates a concise summary of the transcript using the BART model.
- **Keyword Extraction**: Identifies the most relevant keywords or phrases from the transcript.
- **Sentiment Analysis**: Analyzes the sentiment of the transcript and visualizes the sentiment trend over time.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/kumar-017/youtube-video-summarizer.git
    cd youtube-video-summarizer
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```

2. **Enter the YouTube Video Link**
    - Input the YouTube video link in the provided text box.

3. **Generate Outputs**
    - Click on the buttons to generate the summary, key phrases, and sentiment analysis graph.

## Project Structure

- `yt.py`: The main file that runs the Streamlit app.
- `requirements.txt`: Contains the list of dependencies required for the project.
- `youtube_utils.py`: Contains helper functions for summarization, keyword extraction, and sentiment analysis.

## Example

Here is an example of how to use the app:

1. **Summarize Video**
    - Enter the YouTube video link.
    - Click the "Summarize" button to get a concise summary of the video content.

2. **Extract Key Phrases**
    - Click the "Key Phrases" button to get a list of important phrases from the video transcript.

3. **Sentiment Analysis**
    - Click the "Sentiment Analysis" button to generate a sentiment trend graph for the video.

## Limitations
- **Transcript Availability**: The tool relies on the availability of transcripts for the videos. If a transcript is not available, the tool cannot function.
- **Accuracy**: The quality of the summarization and keyword extraction depends on the models used and may not always be perfect.
- **Processing Time**: For long videos, processing can take a significant amount of time.

## Future Enhancements
- **Multi-language Support**: Add support for transcripts in multiple languages.
- **Enhanced Summarization**: Improve the summarization algorithm for better accuracy.
- **Customization**: Allow users to customize the number of key phrases and summary length.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.



---

Thank you for using the YouTube Video Analyzer! We hope this tool helps you save time and gain insights from YouTube videos more efficiently.
