from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # Free model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Hello, I am an AI agent."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

from langchain import OpenAI, ConversationChain

llm = OpenAI(model="text-davinci-003", api_key=None)  # Free tier
conversation = ConversationChain(llm=llm)

response = conversation.run(input="What is AI?")
print(response)
