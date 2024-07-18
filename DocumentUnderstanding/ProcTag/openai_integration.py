import os
from openai import OpenAI

def call_GPT(message_content):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": message_content,}
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content
