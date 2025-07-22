# from util.llm_factory import LLMFactory
from util.system_prompt import prompt_generate_summary
# from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from util.simpleagent import MyAgent
# def generate_report(text):
#     response = LLMFactory.invoke(
#         system_prompt=prompt_generate_summary,
#         human_message=text,
#         temperature=0.7,
#         local_llm=False,
#     )
#     summary = response.content.strip()
#     return summary
def generate_report(text):
    agent = MyAgent(system_prompt=prompt_generate_summary)
    summary = agent(message=text)
    return summary









if __name__ == "__main__":
    Question = input("User Prompt")# or ocr data
    Requirements = input("Job Description")# or ocr data
    if not Question or not Requirements:
        raise ValueError("Both User Prompt and Job Description are required.")
    summary = generate_report("user-profile :"+Question+" & jobdescription :"+Requirements)
    print("\nGenerated Report:")
    print(summary)