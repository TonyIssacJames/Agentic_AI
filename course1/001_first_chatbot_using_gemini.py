import google.generativeai as genai
import os
#from google.colab import userdata

'''
Steps
1 - Open Google AI Studio (API keys)
Go to: https://aistudio.google.com/app/apikey

2 Sign in with your Google account and accept the terms if prompted.

3 -Create a key
Use “Get API key” / “Create API key”. You can attach it to a new Cloud project or an existing one (AI Studio walks you through this).

4 -Copy the key and store it somewhere safe. Don’t commit it to git.

5 - nano ~/.config/secrets.sh
    export GOOGLE_API_KEY="your-real-key-here" # replace with your actual key

6 - source ~/.config/secrets.sh (before running the script)
'''



# do this in the enviorment
# export GOOGLE_API_KEY="your-key-here"

#genai.configure(api_key=userdata.get("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available models")

for m in genai.list_models():
    print(f"{m.name} ({m.supported_generation_methods})")
print("*"*100)
print("*"*100)
# load gemini model
#gemini-2.5-flash-lite
#/gemini-pro-latest  - best model
model = genai.GenerativeModel("gemini-2.5-flash-lite")  

# define a very simple function


def simple_agent(prompt):
    response = model.generate_content(prompt)
    return response.text


# limit the response to 200 words to limit tokens
question = "explain RAG in agentic AI - as you would explain to a 5 year old, limit your response to 200 words"
answer = simple_agent(question)

print("\n\n\nques:\n", question)
print("\n\nans:\n", answer)
