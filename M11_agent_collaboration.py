#%% Packages
import os
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

#%% Manager-LLM
MODEL = 'llama-3.1-70b-versatile'
llm = ChatGroq(model=MODEL,
               temperature=0,
               api_key=os.getenv('GROQ_API_KEY'))


#%% Agents
question_creator = Agent(
    role="Question Creator",
    goal="Create {number} questions for a multiple choice test on {topic}",
    backstory="You are a professional question creator. "
              "You have been tasked to create multiple choice questions for a test on {topic}."
              "Create {number} questions",
    allow_delegation=False,
	verbose=True
)

answer_creator = Agent(
    role="Answer Creator",
    goal="Create answers for the multiple choice questions",
    backstory="You are a professional answer creator. "
              "You have been tasked to create answers for the multiple choice questions."
              "For each answer provide the information correct/incorrect"
              "Also provide the info why the answer is correct or incorrect"
              "Examples: 'B: correct, because ...' 'C: incorrect, because ...'",
    allow_delegation=False,
    verbose=True
)

qa_compiler = Agent(
    role="Create a multiple choice test",
    goal="create a question and answer set for a multiple choice test",
    backstory="You are a professional proof reader. "
              "You have been tasked to proof read the questions and answers."
              "Make sure that the questions are clear and the answers are correct",
    allow_delegation=False,
    verbose=True
)
# %% Tasks
create_questions = Task(
    agent=question_creator,
    description="Create one question for a multiple choice test on {topic}",
    expected_output = "A question created for the multiple choice test on {topic}",
)
#%%
create_answers = Task(
    agent=answer_creator,
    description=("Create {number} answers for the multiple choice questions"),
    expected_output = "{number} answers created for the multiple choice questions",
)
#%%
create_qa = Task(
    agent=qa_compiler,
    description=("create a question and answer set for a multiple choice test"),
    expected_output = ("Question and corresponding answers, "
                       "Example: Question: 'What is the capital of France?' Possible Answers: 'A: Paris', 'B: London', 'C: Berlin', 'D: Madrid'; Rationale: Correct Answer A because ..., Incorrect Answer B because ...")
)
# %% Set up the crew
crew = Crew(
    agents=[question_creator, answer_creator],
    tasks=[create_questions, create_answers, create_qa],
    process=Process.hierarchical,
    manager_llm=llm, 
    manager_agent=qa_compiler,
    planning=True
)

res = crew.kickoff(inputs={'number': '4', 'topic': 'data chunking'})


# %%
pprint(res.raw)
# %%
