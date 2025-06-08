import json
import logging
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import joblib
import pandas as pd
import numpy as np
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_FIELDS = ["event", "vibe", "date", "budget", "time", "duration", "venue", "location", "guests", "special_instructions"]
DECLINE_KEYWORDS = {"no", "none", "nothing", "nothing to say", "no instructions", "nil", "n/a", "na", "nope", "null"}

# Load model components for recommendations
model_components = joblib.load('service_recommender.pkl')
clf = model_components['model']
encoders = model_components['encoders']
tfidf_instructions = model_components['tfidf_instructions']
mlb = model_components['mlb']

def get_event_recommendations(input_data):
    events = pd.read_csv('vega_detailed_csv_files/generated_events.csv')
    events_i = events[events['event_type']==input_data['event']]
    if len(events_i) == 0:
        return None, None
    flexible_budget = input_data['budget'] * 1.1
    budget_filtered_events = events_i[events_i['cost'] <= flexible_budget]
    if len(budget_filtered_events) == 0:
        return None, None
    budget_filtered_events = budget_filtered_events.sort_values('cost', ascending=False)
    selected_event = budget_filtered_events.iloc[0]
    selected_event['services_included'] = ast.literal_eval(selected_event['services_included'])
    return selected_event['package_id'], selected_event

def get_service_recommendations(input_data):
    input_df = pd.DataFrame([input_data])
    input_df['event_id'] = encoders['event_id'].transform(input_df['event'])
    input_df['vibe'] = encoders['vibe'].transform(input_df['vibe'])
    input_df['location'] = encoders['location'].transform(input_df['location'])
    input_df['venue'] = encoders['venue'].transform(input_df['venue'])
    instructions_tfidf_features = tfidf_instructions.transform(input_df['special_instructions'].astype(str))
    X_input = np.hstack([
        input_df[['event_id', 'vibe', 'location', 'budget', 'venue', 'duration', 'guests']].values,
        instructions_tfidf_features.toarray()
    ])
    predictions = clf.predict(X_input)
    prediction_probs = clf.predict_proba(X_input)
    confidence_scores = []
    for i, service in enumerate(mlb.classes_):
        if service.lower() in ['no', 'n', 'o'] or len(service) <= 1:
            continue
        service_probs = prediction_probs[i]
        if isinstance(service_probs, np.ndarray) and len(service_probs.shape) > 1:
            confidence = service_probs[0][1]
        else:
            confidence = service_probs[1] if len(service_probs) > 1 else service_probs[0]
        if confidence > 0.1:
            confidence_scores.append((service, confidence))
    sorted_services = sorted(confidence_scores, key=lambda x: x[1], reverse=True)
    top_services = sorted_services[:5] if len(sorted_services) > 5 else sorted_services
    all_services = pd.read_csv('vega_detailed_csv_files/generated_providers.csv')
    recommendations = []
    for service, confidence in top_services:
        jammed = all_services[all_services['service_id'] == service]
        if not jammed.empty:
            recommendations.append({
                'service_id': service,
                'confidence': confidence,
                'category': jammed['service_category'].values[0],
                'title': jammed['service_title'].values[0],
                'description': jammed['service_description'].values[0]
            })
    return recommendations

class ChatHandler:
    def __init__(self, model="gemma3:1b", streaming=False):
        self.llm = OllamaLLM(model=model, streaming=streaming)
        self.data = {}
        self.current_missing = None
        self.special_asked = False
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
            if k == "special_instructions":
                if isinstance(v, str) and v.strip().lower() in DECLINE_KEYWORDS:
                    self.data[k] = "No"
                elif v not in [None, "null", "Null", ""]:
                    self.data[k] = v
            else:
                if v not in [None, "null", "Null", ""]:
                    self.data[k] = v
    def get_missing_field(self):
        for field in REQUIRED_FIELDS:
            if field == "special_instructions":
                if field not in self.data or self.data[field] in [None, "", "null", "Null"]:
                    if not self.special_asked:
                        return field
            elif field not in self.data or self.data[field] in [None, "", "null", "Null"]:
                return field
        return None
    def handle(self, prompt_template, user_input):
        history = json.dumps(self.data)
        current_field = self.get_missing_field()
        if current_field == "special_instructions":
            self.special_asked = True
        prompt = prompt_template.format(user_input=user_input, history=history, current_field=current_field)
        try:
            raw = self.llm.invoke(prompt)
            parsed = self.parse_response(raw)
            self.update_data(parsed)
            next_missing = self.get_missing_field()
            if next_missing:
                return f"Please provide the following detail: {next_missing}", None
            else:
                return None, self.data
        except Exception as e:
            st.error(f"LLM Error: {e}")
            return "Something went wrong.", None

prompt_template = ChatPromptTemplate.from_template("""
You are an assistant helping extract structured data for an event planner.
The user has provided this so far:
{history}
They just said:
"{user_input}"
The field we're currently trying to fill is: "{current_field}"
Please update the appropriate field based on the user's latest message, and return the full JSON with these keys:
- event
- vibe
- date
- budget
- time
- duration
- venue
- location
- guests
- special_instructions
make sure you get the Date in this format DD-MM-YYYY and time in 24hrs format and remember the present year is 2025
and for duration just get the number of hours only the numerical value
Set any fields not yet provided to null.
Return only the updated JSON object. No explanation.
""")

st.set_page_config(page_title="Event Planner Chat", page_icon="💍")
st.title("💍 Wedding/Event Planner Assistant")

if 'chat_handler' not in st.session_state:
    st.session_state.chat_handler = ChatHandler()
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Tell me about your event...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        reply, completed_json = st.session_state.chat_handler.handle(prompt_template, user_input)
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
    elif completed_json:
        # Convert fields to correct types for recommendation functions
        try:
            completed_json['budget'] = int(completed_json['budget'])
            completed_json['duration'] = int(completed_json['duration'])
            completed_json['guests'] = int(completed_json['guests'])
        except Exception:
            pass
        # Get recommendations
        service_recs = get_service_recommendations(completed_json)
        event_id, event_data = get_event_recommendations(completed_json)
        # Display recommendations
        rec_text = """\n**Here are your personalized recommendations:**\n"""
        if service_recs:
            rec_text += "\n### Top Service Recommendations:\n"
            for rec in service_recs:
                rec_text += f"- **{rec['title']}** ({rec['category']})\n  - {rec['description']}\n  - Confidence: {rec['confidence']:.2%}\n"
        else:
            rec_text += "\nNo service recommendations found.\n"
        if event_data is not None:
            rec_text += f"\n### Event Package Recommendation:\n- **Event Type:** {event_data['event_type']}\n- **Budget:** ₹{event_data['cost']:,}\n- **Services Included:**\n"
            for i, svc in enumerate(event_data['services_included']):
                rec_text += f"  - {svc['title']} ({svc['category']}): {svc['description']}\n"
        else:
            rec_text += "\nNo event package recommendation found.\n"
        st.session_state.messages.append({"role": "assistant", "content": rec_text})
        st.rerun() 