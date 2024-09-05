import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_utils import summarize_youtube_video,get_average_sentiment_per_minute,  create_sentiment_graph, create_moving_average_graph, extract_key_phrases
import pandas as pd
import numpy as np
from scipy.stats import linregress

def main():
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: black;
            color: white;
            padding: 8px 20px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-right: 10px;
        }
        .stTextInput>div>div>input {
            background-color: black ;
            border: 1px solid white;
            border-radius: 4px;
            padding: 8px;
            font-size: 16px;
            color: white;
        }
        .stMarkdown {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('YouTube Video Analysis Dashboard')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    
    youtube_video_link = st.text_input('Enter YouTube Video Link', '')
    col1, col2, col3 = st.columns(3)
    @st.cache_data(ttl=None)
    def summarize_video(video_link):
        video_id=video_link.split('=')[1]
        txt=''
        yt=YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
        for i in yt:
            txt+=i['text']+' '
        
        summarization = summarize_youtube_video(youtube_video_link)
       
        return summarization
    @st.cache_data(ttl=None)
    def extract_phrases_from_video(video_link):
        video_id=video_link.split('=')[1]
        txt=''
        yt=YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
        for i in yt:
            txt+=i['text']+' '
        
        important_phrases = extract_key_phrases(txt, num_phrases=10)
        
        return important_phrases
    @st.cache_data(ttl=None)
    def analyze_sentiment(video_link):
        video_id=video_link.split('=')[1]
        txt=''
        yt=YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
        for i in yt:
            txt+=i['text']+' '
        average_sentiments_per_minute=get_average_sentiment_per_minute(yt)
        
        
        minutes = list(average_sentiments_per_minute.keys())
        sentiment_scores = list(average_sentiments_per_minute.values())
        slope, intercept, _, _, _ = linregress(minutes, sentiment_scores)
        trend_line = [slope * minute + intercept for minute in minutes]

       
        df = pd.DataFrame({'Minutes': minutes, 'Sentiment Scores': sentiment_scores , 'Trend Line':trend_line })
        st.subheader('Average Sentiment Score per Minute')

       
        
        
        st.line_chart(data=df,x="Minutes")

        

        
        


    if col1.button('Summarize'):
        summarization = summarize_video(youtube_video_link)
        st.subheader('Summarized Text:')
        st.write(summarization)

       

        
    if col2.button('keyphrases'):
        important_phrases = extract_phrases_from_video(youtube_video_link)

        st.subheader('Important Phrases:')
        st.write(important_phrases)
    if col3.button('Sentiment Analysis'):
        

       
        sentiment_plot= analyze_sentiment(youtube_video_link)

        
        
        sentiment_plot

        




if __name__ == '__main__':
    main()