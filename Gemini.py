import google.generativeai as genai

genai.configure(api_key="")
model = genai.GenerativeModel('gemini-2.5-pro')
response = model.generate_content(input("Ask me anything: "))
print(response.text)
