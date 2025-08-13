from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

#step 1a

video_id = "Gfr50f6ZBvo"

try:
    # if you dont care which language
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages = ["en"])

    # flatten it to plain text
    #transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    print("No Caption Available for this Video")

except Exception as e:
     print("An error occurred:", e)



# try:
#     transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    
#     try:
#         transcript = transcript_list.find_manually_created_transcript(['en'])
#     except NoTranscriptFound:
#         transcript = transcript_list.find_generated_transcript(['en'])
    
#     final_transcript = transcript.fetch()
    
#     for entry in final_transcript:
#         print(entry['text'])

# except TranscriptsDisabled:
#     print("No Captions Available for this video")

# except Exception as e:
#     print("An error occurred:", e)
