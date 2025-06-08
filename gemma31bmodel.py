import json
import logging
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_FIELDS = ["Event", "Vibe", "Date","Budget","Time", "Duration","Venue", "Location", "Guests"]


class ChatHandler:
    def __init__(self, model="gemma3:1b", streaming=False):
        self.llm = OllamaLLM(model=model, streaming=streaming)
        self.data = {}
        self.current_missing = None

    def parse_response(self, raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                fixed = raw[raw.find("{"):raw.rfind("}") + 1]
                return json.loads(fixed)
            except:
                return {}

    def update_data(self, parsed):
        for k, v in parsed.items():
            if v not in [None, "null", "Null", ""]:
                self.data[k] = v

    def get_missing_field(self):
        for field in REQUIRED_FIELDS:
            if field not in self.data or self.data[field] in [None, "", "null", "Null"]:
                return field
        return None

    def handle(self, prompt_template, user_input):
        history = json.dumps(self.data)
        current_field = self.get_missing_field()

        prompt = prompt_template.format(user_input=user_input, history=history, current_field=current_field)

        try:
            raw = self.llm.invoke(prompt)
            parsed = self.parse_response(raw)
            self.update_data(parsed)

            next_missing = self.get_missing_field()
            if next_missing:
                return f"Please provide the following detail: {next_missing}"
            else:
                return f"🎉 Here's your completed plan:\n```json\n{json.dumps(self.data, indent=2)}\n```"

        except Exception as e:
            st.error(f"LLM Error: {e}")
            return "Something went wrong."


# -------------------------------------
# Prompt Template
# -------------------------------------
prompt_template = ChatPromptTemplate.from_template("""
You are an assistant helping extract structured data for an event planner.

The user has provided this so far:
{history}

They just said:
"{user_input}"

The field we're currently trying to fill is: "{current_field}"

Please update the appropriate field based on the user's latest message, and return the full JSON with these keys:
- Event
- Vibe
- Date
- Budget
- Time
- Duration
- Venue
- Location
- Guests

make sure you get the Date in this format DD-MM-YYYY and time in 24hrs format and remember the present year is 2025

and for duration just get the number of hours only the numerical value

Set any fields not yet provided to null.
Return only the updated JSON object. No explanation.
""")


# -------------------------------------
# Streamlit UI
# -------------------------------------
st.set_page_config(page_title="Event Planner Chat", page_icon="💍")
st.title("💍 Wedding/Event Planner Assistant")

if 'chat_handler' not in st.session_state:
    st.session_state.chat_handler = ChatHandler()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Tell me about your event...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        reply = st.session_state.chat_handler.handle(prompt_template, user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
