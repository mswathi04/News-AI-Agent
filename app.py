import streamlit as st
from crewai import Crew, Agent, Task, Process
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING,Any,Dict,Optional
from tools import tool

from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os
## call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

avators = {"Writer":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "Reviewer":"https://cdn-icons-png.freepik.com/512/9408/9408201.png"}


class MyCustomHandler(BaseCallbackHandler):

    
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs['input'])
   
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(outputs['output'])

# Agents
news_researcher=Agent(
    role="Senior Researcher",
    goal='Uncover ground breaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world"
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=True,
    callbacks=[MyCustomHandler("Reviewer")],
)

##creating a writer agent with custom tools responsible in writing news blog

news_writer=Agent(
    role='Writer',
    goal='Narrate compelling tech stories about {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=False,
    callbacks=[MyCustomHandler("Writer")],
)



st.title("News AI Agent")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What News do you want us to write?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

st.markdown("""
    <style>
    .chat-message-user {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

if prompt := st.chat_input():

    # Appending the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    user_message_container = st.container()
    with user_message_container:
        st.chat_message("user").write(f'<div class="chat-message-user">{prompt}</div>', unsafe_allow_html=True)


    # Tasks
    research_task=Task(
        description=(
            "Identify the next big trend in {topic}."
            "Focus on identifying pros and cons and the overall narrative."
            "Your final report should learly articulate the key points,"
            "its market opportutites, and the potential risks."
        ),
        expected_output='A comprehensive 3 paragraphs long report on {topic}',
        tools=[tool],
        agent=news_researcher,
    )

    #Writing task with language model comfiguraion
    write_task=Task(
        description=(
            "Compose an insightful article on {topic}."
            "Focus on the latest trends and how it's impacting the industry."
            "This article should be easy to understand, engaging, and positive."
        ),
        expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
        tools=[tool],
        agent=news_writer,
        async_execution=False,
        output_file='new-blog-post.md'
    )

    crew=Crew(
        agents=[news_researcher,news_writer],
        tasks=[research_task,write_task],
        process=Process.sequential
    )

    
    final=crew.kickoff(inputs={'topic':{prompt}})
    print(final)
    result = f"## Here is the Final Result \n\n {final}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)