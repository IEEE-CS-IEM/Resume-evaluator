import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class MyAgent:
    def __init__(self,system_prompt):
        self.system_prompt = system_prompt
        self.message=[] 
    
    def __call__(self,message,temperature=0.3): 
        if (not message or message.strip() == ""):
            raise ValueError("Message cannot be empty")
        self.message.append({"parts":[message]})
        return self.execute(self.message)
    def createClient(self):
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        return genai.GenerativeModel(model_name='gemini-1.5-flash')

    def execute(self, message):
        self.client = self.createClient()
        if message[0].get("parts")[0]:
            return self.client.generate_content(message)
    


