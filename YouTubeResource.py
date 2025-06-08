import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from composio_langchain import ComposioToolSet, Action
import fal_client
from agents import Agent, Runner
import asyncio
from openai import OpenAI
import nest_asyncio
nest_asyncio.apply()

st.set_page_config(page_title="YouTube Resource Generator")
st.title("YouTube Resource Generator")

# User input
user_prompt = st.text_input("Enter your topic (e.g., NCERT Class 10 Science Chapter 1):", "NCERT Class 10 Science Chapter 1")



if st.button("Generate Resources"):
    
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    crewllm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    st.write("CrewAI Section")
    agent = Agent(name="Assistant", instructions="You are a helpful teaching assistant. You are given a topic and you need to find the information and summarize it in a way that is easy to understand for a class 10 student. Summarize the important topics and sub topics. You only go to the original website to find the information. Whereever applicable quote chemical reactions, diagrams, examples")
    st.write("Agent")
    result_oa = Runner.run(agent, user_prompt)
    st.write("result_oa")
    st.subheader("OpenAI Agent Response")
    st.write(result_oa.final_output)

    # MCP (YouTube Search) Section
    prompt = hub.pull("hwchase17/openai-functions-agent")
    composio_toolset = ComposioToolSet(api_key=st.secrets["COMPOSIO_API_KEY"], entity_id="kaushiktest91%gmail.com")

    def filter_youtube_results(result: dict) -> dict:
        if not result.get("successful") or "data" not in result:
            st.write("No results found")
            return result
        original_messages = result["data"].get("messages", [])
        st.write(original_messages)
        if not isinstance(original_messages, list):
            st.write("No results found2")
            return result
        filtered_results = []
        st.write("filtered_results")
        for result in original_messages:
            st.write(result)
            filtered_results.append({
                "title": result["data"]["response_data"]["items"]["snippet"]["title"],
                "description": result["data"]["response_data"]["items"]["snippet"]["description"]
            })
        processed_result = {
            "successful": True,
            "data": {"summary": filtered_results},
            "error": None
        }
        st.write(processed_result)
        return processed_result

    processed_tools = composio_toolset.get_tools(
        actions=[Action.YOUTUBE_SEARCH_YOU_TUBE],
        processors={
            "post": {Action.YOUTUBE_SEARCH_YOU_TUBE: filter_youtube_results}
        }
    )
    st.write(processed_tools)
    st.write("processed_tools")
    agent2 = create_openai_functions_agent(crewllm, processed_tools, prompt)
    agent_executor = AgentExecutor(agent=agent2, tools=processed_tools, verbose=False)
    task = f"Go to youtube and search for {user_prompt} and return details of the video that appear in the search results."
    composio_result = agent_executor.invoke({"input": task})
    st.subheader("Composio (YouTube Search) Response")
    st.json(composio_result)

    # Text-to-Audio Section
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
    def on_queue_update(update):
        pass  # No logs in Streamlit
    # Use the OpenAI agent's output as the text for TTS if possible
    chapter_summary_text = str(result_oa) if result_oa else user_prompt
    audio_result = fal_client.subscribe(
        "fal-ai/elevenlabs/tts/turbo-v2.5",
        arguments={
            "text": chapter_summary_text
        },
        with_logs=False,
        on_queue_update=on_queue_update,
    )
    audio_url = audio_result.get("audio_url") or audio_result.get("url")
    st.subheader("fal AI (Text-to-Audio) Response")
    if audio_url:
        st.audio(audio_url)
    else:
        st.write("Audio generation failed or no audio URL returned.")

