import streamlit as st
import os
import json
import asyncio
import nest_asyncio
import base64
nest_asyncio.apply()
import dotenv
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from composio_langchain import ComposioToolSet, Action
import fal_client
from agents import Agent, Runner
from openai import OpenAI
from youtubesearchpython import VideosSearch

st.set_page_config(page_title="YouTube Resource Generator")
st.title(":red[YouTube Resource Generator]")
st.write("This tool will help new YouTube content creators to generate all the resources required for creating their first YouTube videos. The output will have Text Content, Audio File, Competitor Analysis, Thumbnail and MCQs on the topic. This video generator is focused on CBSE topics.")

# User input

classes=st.selectbox("Select the Class", ("Class 9", "Class 10", "Class 11", "Class 12"))
subject=st.selectbox("Select the Subject", ("Science", "Maths", "Sociology", "History", "Geography", "Political Science", "Economics"))
chapter=st.text_input("Enter the Chapter Number", "1")
user_prompt = f"NCERT {classes} {subject} Chapter {chapter}"
st.write(f"Topic: {user_prompt}")



if st.button("Generate Resources"):
    try:
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
        # CrewAI Section
        with st.spinner("Generating summary with AI Agents..."):
            crewllm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = Agent(name="Assistant", instructions="You are a helpful teaching assistant. You are given a topic and you need to find the information and summarize it in a way that is easy to understand for students. Summarize the important topics and sub topics. You only go to the original NCERT website to find the information. Whereever applicable quote chemical reactions, equations, diagrams, examples.")
            
            # Handle the async function properly
            result_oa = asyncio.run(Runner.run(agent, user_prompt))
            
            st.subheader(":blue[Contents of the Video for the given topic:]")
            st.write(result_oa.final_output)
    except Exception as e:
        st.error(f"Error in AI Agents section: {str(e)}")

    # Text-to-Audio Section
    try:
        with st.spinner("Generating audio from text..."):
            # Check if FAL_KEY is available
            fal_key = os.getenv("FAL_KEY")
            if not fal_key:
                st.error("FAL_KEY environment variable is not set. Please check your .env file.")
                st.info("To fix this, add your FAL_KEY to the .env file in the format: FAL_KEY=your_key_here")
                raise ValueError("Missing FAL_KEY environment variable")
            
            os.environ["FAL_KEY"] = fal_key
            
            # Use the OpenAI agent's output as the text for TTS
            chapter_summary_text = str(result_oa.final_output) if result_oa else user_prompt
            
            # Limit text length if needed (ElevenLabs has a character limit)
            max_chars = 2500  # ElevenLabs turbo has lower limits
            if len(chapter_summary_text) > max_chars:
                st.warning(f"Text too long ({len(chapter_summary_text)} chars), truncating to {max_chars} chars")
                chapter_summary_text = chapter_summary_text[:max_chars] + "..."
            
            st.subheader(":blue[Audio for the contents of the video:]")
            st.write(f"Generating audio for text ({len(chapter_summary_text)} chars)")
            
            # Try both parameter formats to handle API changes
            try:
                # First try with 'voice' parameter
                audio_result = fal_client.subscribe(
                    "fal-ai/elevenlabs/tts/turbo-v2.5",
                    arguments={
                        "text": chapter_summary_text,
                        "voice": '21m00Tcm4TlvDq8ikWAM'  # Rachel voice
                    },
                    with_logs=True
                )
            except Exception as e:
                st.warning(f"First attempt failed: {str(e)}. Trying alternative parameter format...")
                # Try with 'voice_id' parameter
                audio_result = fal_client.subscribe(
                    "fal-ai/elevenlabs/tts/turbo-v2.5",
                    arguments={
                        "text": chapter_summary_text,
                        "voice_id": '21m00Tcm4TlvDq8ikWAM'  # Rachel voice
                    },
                    with_logs=True
                )
            
           
            # Get the audio URL
            audio_url = audio_result.get("audio_url") or audio_result.get("url")
            
            if audio_url:
                st.success("Audio generation successful!")
                st.write(audio_url)
            else:
                st.error("Audio generation failed.")

    except Exception as e:
        st.error(f"Error in Text-to-Audio section: {str(e)}")

        
        # Fallback to just displaying the text if audio generation fails




    # YouTube Search Section (using youtube-search-python)
    try:
        with st.spinner("Searching for videos..."):
            # Direct YouTube search without API limits
            search_query = f"{user_prompt}"
            st.write(f"Searching YouTube for: {user_prompt}")
            
            # Perform the search
            videos_search = VideosSearch(search_query, limit=5)
            search_results = videos_search.result()
            
            # Display the raw results for debugging
          
            # Process the results
            videos = []
            if 'result' in search_results and len(search_results['result']) > 0:
                for video in search_results['result']:
                    videos.append({
                        "title": video.get('title', 'No title'),
                        "link": video.get('link', ''),
                        "duration": video.get('duration', 'Unknown'),
                        "channel": video.get('channel', {}).get('name', 'Unknown channel'),
                        "views": video.get('viewCount', {}).get('short', 'Unknown views'),
                        "published": video.get('publishedTime', 'Unknown'),
                        "description": video.get('descriptionSnippet', 'Unknown description'),
                        "thumbnail": video.get('thumbnails', [{}])[0].get('url', '') if video.get('thumbnails') else ''
                    })
                
                st.subheader(f":blue[Top {len(videos)} competitor videos for the topic]")
                
                # Display videos in a more user-friendly way
                for i, video in enumerate(videos):
                    with st.expander(f"Video {i+1}: {video['title']}"):
                        # Display thumbnail if available
                        if video['thumbnail']:
                            st.image(video['thumbnail'], use_container_width=True)
                        
                        # Display video details
                        st.write(f"**Channel:** {video['channel']}")
                        st.write(f"**Views:** {video['views']}")
                        st.write(f"**Duration:** {video['duration']}")
                        st.write(f"**Published:** {video['published']}")
                        st.write(f"**Description:** {video['description']}")
                        st.write(f"**Link:** [{video['link']}]({video['link']})")
                        
                        # Add a button to watch the video
                        if st.button(f"Watch Video {i+1}", key=f"watch_{i}"):
                            st.video(video['link'])
            else:
                st.warning("No videos found for your search query.")
    except Exception as e:
        st.error(f"Error in YouTube search section: {str(e)}")
        st.write("Exception details:", type(e).__name__, str(e))


    # Thumbnail Section
    try:
        with st.spinner("Generating thumbnail from topic..."):
            client = OpenAI()
            img = client.images.generate(
                model="dall-e-3",
                prompt=f"Generate a Thumbnail image with the following: 1. Youtube icon in the cennter of the image, 2. Chapter name from  {user_prompt} on the top of the image and 3. Only 1 diagram reelevant to {chapter_summary_text} on the lower part of the image ",
                n=1,
                size="1024x1024"
            )
        
        st.subheader(":blue[Thumbnail for the video:]")
        st.image(img.data[0].url)
    except Exception as e:
        st.error(f"Error in Thumbnail section: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
