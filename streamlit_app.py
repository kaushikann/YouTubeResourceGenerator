import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from composio_langchain import ComposioToolSet, Action
import fal_client
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(page_title="YouTube Resource Generator")
st.title("YouTube Resource Generator")

# User input
user_prompt = st.text_input("Enter your topic (e.g., NCERT Class 10 Science Chapter 1):", "NCERT Class 10 Science Chapter 1")

if st.button("Generate Resources"):
    # CrewAI Section
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    crewllm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent1 = Agent(role="search agent", goal=user_prompt, backstory="You are a teacher and you are given a topic. You need to find the information and summarize it in a way that is easy to understand for a class 10 student. Summarize the important topics and sub topics. You only go to the original website to find the information. Whereever applicable quote chemical reactions, diagrams, examples",)
    task1 = Task(description="search the internet and accomplish the goal", expected_output="Explanation of the topic and subtopic with examples, diagrams, reactions, etc. The source of information and any PDF links", agent=agent1)
    crew1 = Crew(agents=[agent1], tasks=[task1], verbose=True)
    chapter_summary = crew1.kickoff()
    chapter_summary_text = str(chapter_summary)
    st.subheader("CrewAI Output")
    st.write(chapter_summary_text)

    # MCP (YouTube Search) Section
    prompt = hub.pull("hwchase17/openai-functions-agent")
    composio_toolset = ComposioToolSet(api_key=st.secrets["COMPOSIO_API_KEY"], entity_id="kaushiktest91%gmail.com")

    def filter_youtube_results(result: dict) -> dict:
        if not result.get("successful") or "data" not in result:
            return result
        original_messages = result["data"].get("messages", [])
        if not isinstance(original_messages, list):
            return result
        filtered_results = []
        for result in original_messages:
            filtered_results.append({
                "title": result.get("title", ""),
                "description": result.get("description", ""),
            })
        processed_result = {
            "successful": True,
            "data": {"summary": filtered_results},
            "error": None
        }
        return processed_result

    processed_tools = composio_toolset.get_tools(
        actions=[Action.YOUTUBE_SEARCH_YOU_TUBE],
        processors={
            "post": {Action.YOUTUBE_SEARCH_YOU_TUBE: filter_youtube_results}
        }
    )
    agent = create_openai_functions_agent(crewllm, processed_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=processed_tools, verbose=False)
    task = f"Go to youtube and search for {user_prompt} and return details of the video that appear in the search results."
    result = agent_executor.invoke({"input": task})
    st.subheader("MCP (YouTube Search) Response")
    st.json(result)

    # Text-to-Audio Section
    os.environ["FAL_KEY"] = st.secrets["FAL_KEY"]
    def on_queue_update(update):
        pass  # No logs in Streamlit
    audio_result = fal_client.subscribe(
        "fal-ai/elevenlabs/tts/turbo-v2.5",
        arguments={
            "text": chapter_summary_text
        },
        with_logs=False,
        on_queue_update=on_queue_update,
    )
    audio_url = audio_result.get("audio_url") or audio_result.get("url")
    st.subheader("Audio Output")
    if audio_url:
        st.audio(audio_url)
    else:
        st.write("Audio generation failed or no audio URL returned.") 
