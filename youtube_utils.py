from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration
import re
import torch
from keybert import KeyBERT






def summarize_youtube_video(youtube_video_link):
    video_id=youtube_video_link.split('=')[1]
    txt=''
    yt=YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
    for i in yt:
        txt+=i['text']+ ' '
    
    
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    
    def generate_summary(input_text, max_chunk_length=1000, max_summary_length=200):
    
        input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
        chunks = [input_tokens[i:i + max_chunk_length] for i in range(0, len(input_tokens), int(max_chunk_length //1.5))]

        
        summary = ""
        for chunk_tokens in chunks:
            inputs = tokenizer.build_inputs_with_special_tokens(chunk_tokens)
            inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)
            summary_ids = model.generate(inputs, max_length=max_summary_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "

        return summary.strip()
    summary=generate_summary(txt)
    summary = re.sub(r'(?<=\w)([.!?"])(?=\w)', r'\1 ', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    

    
    return summary


def extract_key_phrases(input_text, num_phrases=10):
    
    
    
    model = KeyBERT()
    key_phrases = model.extract_keywords(input_text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=num_phrases)
    key_phrases = [phrase[0] for phrase in key_phrases]

    return key_phrases
import stanza
stanza.download('en')
nlp=stanza.Pipeline()
def calculate_sentiment_score(text):
    
    d=nlp(text)
    sentiment_score =0
    
    for sent in d.sentences:
        sentiment_score += sent.sentiment
    return sentiment_score
def get_average_sentiment_per_minute(transcript_data):
    
    sentiments_per_minute = {}
    

    for item in transcript_data:
        text = item['text']
        start_time = item['start']
        duration = item['duration']

        
        sentiment_score = calculate_sentiment_score(text)
        
        end_time = start_time + duration


        start_minute = int((start_time // 60)+1)
        end_minute = int((end_time // 60)+1)


        for minute in range(start_minute, end_minute + 1):
            if minute in sentiments_per_minute:
                sentiments_per_minute[minute].append(sentiment_score)
            else:
                sentiments_per_minute[minute] = [sentiment_score]


    average_sentiments = {}
    for minute, scores in sentiments_per_minute.items():
        average_sentiments[minute] = sum(scores) / len(scores)

    return average_sentiments


import matplotlib.pyplot as plt
def create_sentiment_graph(average_sentiments_per_minute):
    
    
    minutes = list(average_sentiments_per_minute.keys())
    sentiment_scores = list(average_sentiments_per_minute.values())

    
    plt.plot(minutes, sentiment_scores, marker='o', linestyle='-')
    plt.xlabel('Minutes')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score per Minute')
    plt.grid(True)
    plt.show()
import numpy as np
def create_moving_average_graph(average_sentiments_per_minute):
    
    
    
    
    sentiment_scores = list(average_sentiments_per_minute.values())

    window_size = 3  
    moving_average = np.convolve(sentiment_scores, np.ones(window_size)/window_size, mode='valid')
    start_time_offset = (window_size // 2)  
    minutes_with_moving_average = range(start_time_offset, start_time_offset + len(moving_average))

    plt.plot(minutes_with_moving_average, moving_average, color='red', label='Moving Average (Window Size 3)')
    plt.legend()
