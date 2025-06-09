import streamlit as st
import os
import json
import asyncio
import nest_asyncio
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
st.title("YouTube Resource Generator")

# User input
user_prompt = st.text_input("Enter your topic (e.g., NCERT Class 10 Science Chapter 1):", "NCERT Class 10 Science Chapter 1")



if st.button("Generate Resources"):
    try:
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
        # CrewAI Section
        with st.spinner("Generating summary with CrewAI..."):
            crewllm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = Agent(name="Assistant", instructions="You are a helpful teaching assistant. You are given a topic and you need to find the information and summarize it in a way that is easy to understand for a class 10 student. Summarize the important topics and sub topics. You only go to the original website to find the information. Whereever applicable quote chemical reactions, diagrams, examples")
            
            # Handle the async function properly
            result_oa = asyncio.run(Runner.run(agent, user_prompt))
            
            st.subheader("OpenAI Agent Response")
            st.write(result_oa)
    except Exception as e:
        st.error(f"Error in CrewAI section: {str(e)}")

    # YouTube Search Section (using youtube-search-python)
    try:
        with st.spinner("Searching for videos..."):
            # Direct YouTube search without API limits
            search_query = f"{user_prompt} tutorial"
            st.write(f"Searching YouTube for: {search_query}")
            
            # Perform the search
            videos_search = VideosSearch(search_query, limit=5)
            search_results = videos_search.result()
            
            # Display the raw results for debugging
            with st.expander("Raw YouTube Search Results"):
                st.json(search_results)
            
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
                        "thumbnail": video.get('thumbnails', [{}])[0].get('url', '') if video.get('thumbnails') else ''
                    })
                
                st.subheader(f"Found {len(videos)} Videos for '{user_prompt}'")
                
                # Display videos in a more user-friendly way
                for i, video in enumerate(videos):
                    with st.expander(f"Video {i+1}: {video['title']}"):
                        # Display thumbnail if available
                        if video['thumbnail']:
                            st.image(video['thumbnail'], use_column_width=True)
                        
                        # Display video details
                        st.write(f"**Channel:** {video['channel']}")
                        st.write(f"**Views:** {video['views']}")
                        st.write(f"**Duration:** {video['duration']}")
                        st.write(f"**Published:** {video['published']}")
                        st.write(f"**Link:** [{video['link']}]({video['link']})")
                        
                        # Add a button to watch the video
                        if st.button(f"Watch Video {i+1}", key=f"watch_{i}"):
                            st.video(video['link'])
            else:
                st.warning("No videos found for your search query.")
                
                # Fallback: Create mock results for testing UI
                st.subheader("Sample Videos (Mock Data)")
                mock_videos = [
                    {
                        "title": f"{user_prompt}: Complete Tutorial",
                        "link": "https://www.youtube.com/watch?v=sample1",
                        "duration": "10:24",
                        "channel": "Tech Tutorials",
                        "views": "1.2M views",
                        "published": "2 months ago"
                    },
                    {
                        "title": f"Learn {user_prompt} in 30 Minutes",
                        "link": "https://www.youtube.com/watch?v=sample2",
                        "duration": "15:36",
                        "channel": "Learning Made Easy",
                        "views": "850K views",
                        "published": "1 year ago"
                    }
                ]
                for i, video in enumerate(mock_videos):
                    with st.expander(f"Sample Video {i+1}: {video['title']}"):
                        st.write(f"**Channel:** {video['channel']}")
                        st.write(f"**Views:** {video['views']}")
                        st.write(f"**Duration:** {video['duration']}")
                        st.write(f"**Published:** {video['published']}")
                        st.write(f"**Note:** This is sample data since the search returned no results")
    except Exception as e:
        st.error(f"Error in YouTube search section: {str(e)}")
        st.write("Exception details:", type(e).__name__, str(e))
        import traceback
        st.code(traceback.format_exc())

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
            chapter_summary_text = str(result_oa) if result_oa else user_prompt
            
            # Limit text length if needed (ElevenLabs has a character limit)
            max_chars = 2500  # ElevenLabs turbo has lower limits
            if len(chapter_summary_text) > max_chars:
                st.warning(f"Text too long ({len(chapter_summary_text)} chars), truncating to {max_chars} chars")
                chapter_summary_text = chapter_summary_text[:max_chars] + "..."
            
            st.subheader("Text-to-Audio Result")
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
            
            # Debug the result
            with st.expander("API Response Details"):
                st.write("Audio result keys:", list(audio_result.keys()))
                st.json(audio_result)
            
            # Get the audio URL
            audio_url = audio_result.get("audio_url") or audio_result.get("url")
            
            if audio_url:
                st.success("Audio generation successful!")
                st.audio(audio_url)
            else:
                st.error("Audio generation failed.")
                st.json(audio_result)  # Show the full response for debugging
    except Exception as e:
        st.error(f"Error in Text-to-Audio section: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("To fix FAL API issues, make sure your FAL_KEY is correct and that you have credits available on your fal.ai account.")
        
        # Fallback to just displaying the text if audio generation fails
        st.subheader("Summary Text (Audio generation failed)")
        st.write(str(result_oa) if result_oa else user_prompt)

