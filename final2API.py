import ast
import joblib
import numpy as np
import pandas as pd # type: ignore
import streamlit as st
import json
from openai import OpenAI
import time

# Initialize OpenAI client
client = OpenAI(
    api_key="<custom api key>",
    base_url="https://api.sambanova.ai/v1",
)

# Load model components
model_components = joblib.load('service_recommender.pkl')
clf = model_components['model']
encoders = model_components['encoders']
tfidf_instructions = model_components['tfidf_instructions']
mlb = model_components['mlb']

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

def check_null_values(data):
    null_fields = []
    for key, value in data.items():
        if value is None:
            null_fields.append(key)
    return null_fields

def get_package_recommendations(input_data):
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

def get_event_recommendations(input_data):
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=[{"role":"user","content":[{"type":"text","text":f"{input_data}"}]}],
        temperature=0.1,
        top_p=0.1
    )
    st.session_state.history.append(response.choices[0].message.content)
    return response.choices[0].message.content

# Set page config
st.set_page_config(
    page_title="Event Planning Assistant",
    page_icon="🎉",
    layout="wide"
)

# Title and description
st.title("🎉 Event Planning Assistant")
st.markdown("""
This assistant helps you plan your event by collecting and organizing all the necessary details.
Simply describe your event or provide specific details, and the assistant will help structure the information.
""")

# Initialize session state for event data
if 'event_data' not in st.session_state:
    st.session_state.event_data = {
        "event": None,
        "vibe": None,
        "date": None,
        "budget": None,
        "time": None,
        "duration": None,
        "venue": None,
        "location": None,
        "guests": None,
        "special_instructions": None
    }

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Input area
    st.subheader("Tell us about your event")
    user_input = st.text_area("Enter your event details or requirements:", height=150)
    user_input = user_input.lower()
    if st.button("Process Information"):
        if user_input:
            with st.spinner("Processing your request..."):
                input_data = f"""You are an assistant helping extract structured data for an event planner.
                    The user has provided this so far:
                    {st.session_state.history}
                    They just said:
                    "{user_input}"
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
                    Return only the updated JSON object. No explanation."""
                
                result = get_event_recommendations(input_data)
                json_data = json.loads(result[7:-3])
                st.session_state.event_data = json_data
                
                # Check for null values
                null_fields = check_null_values(json_data)
                if null_fields:
                    st.warning("Please provide the following missing information:")
                    for field in null_fields:
                        value = st.text_input(f"Enter {field}:", key=f"input_{field}")
                        if value:
                            st.session_state.event_data[field] = value
                
                # Get service recommendations
                if not null_fields:
                    with st.spinner("Getting service recommendations..."):
                        service_recommendations = get_service_recommendations(st.session_state.event_data)
                        package_id, package_details = get_package_recommendations(st.session_state.event_data)
                        
                        # Display package recommendations
                        if package_id and package_details is not None:
                            st.write("Services Included:")
                            for service in package_details['services_included']:
                                st.write(f"- {service}")
                            st.subheader("📦 Recommended Package")
                            st.write(f"Package ID: {package_id}")
                            st.write(f"Cost: ${package_details['cost']:,.2f}")
                            
                        
                        # Display service recommendations
                        if service_recommendations:
                            st.subheader("🔍 Recommended Services")
                            for service in service_recommendations:
                                with st.expander(f"{service['title']} ({service['category']}) - Confidence: {service['confidence']:.2%}"):
                                    st.write(service['description'])
        else:
            st.warning("Please enter some information about your event.")

with col2:
    # Display current event details
    st.subheader("Current Event Details")
    if st.session_state.event_data:
        event_data = st.session_state.event_data
        for key, value in event_data.items():
            if value is not None:
                st.metric(label=key.replace("_", " ").title(), value=value)
            else:
                st.metric(label=key.replace("_", " ").title(), value="Not specified")

# Add a download button for the event details
if st.session_state.event_data:
    json_str = json.dumps(st.session_state.event_data, indent=2)
    st.download_button(
        label="Download Event Details",
        data=json_str,
        file_name="event_details.json",
        mime="application/json"
    )

# Add a reset button
if st.button("Reset Event Details"):
    st.session_state.event_data = {
        "event": None,
        "vibe": None,
        "date": None,
        "budget": None,
        "time": None,
        "duration": None,
        "venue": None,
        "location": None,
        "guests": None,
        "special_instructions": None
    }
    st.session_state.history = []
    st.experimental_rerun() 





    
