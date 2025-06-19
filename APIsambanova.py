import ast
import json
import joblib
import numpy as np
from openai import OpenAI
import pandas as pd

client = OpenAI(
    api_key="<YOUR API KEY>",
    base_url="https://api.sambanova.ai/v1",
)



model_components = joblib.load('service_recommender.pkl')
clf = model_components['model']
encoders = model_components['encoders']
tfidf_instructions = model_components['tfidf_instructions']
mlb = model_components['mlb']





history = []

def check_null_values(data):
    null_fields = []
    for key, value in data.items():
        if value is None:
            null_fields.append(key)
    return null_fields

def get_event_recommendations(input_data):
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=[{"role":"user","content":[{"type":"text","text":f"{input_data}"}]}],
        temperature=0.1,
        top_p=0.1
    )
    history.append(response.choices[0].message.content)
    return response.choices[0].message.content


def get_package_recommendations(input_data):
    events = pd.read_csv('vega_detailed_csv_files/generated_events.csv')
    events_i = events[events['event_type']==input_data['event']]
    if len(events_i) == 0:
        return None, None
    flexible_budget = float(input_data['budget']) * 1.1
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




while True:
    query = input("Enter your query: ")
    input_data = f"""You are an assistant helping extract structured data for an event planner.
                The user has provided this so far:
                {history}
                They just said:
                "{query}"
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
                i dont need any code or explanations for updating or altering i just need only the json response as i desired
                if any null values are present, then ask the user to provide the missing information
                gimme all the data only in lower case letters
                Return only the updated JSON object. No explanation."""
    result = get_event_recommendations(input_data)
    print(result)
    try:
        json_data = json.loads(result)
    except json.JSONDecodeError as e:
        json_data = json.loads(result)
        print(f"failed in decoding {e}")
        print(f"raw response {result}")
    print("\nCurrent event details:")
    print(json.dumps(json_data, indent=2))
    
    # Check for null values and ask for them one by one
    null_fields = check_null_values(json_data)
    if null_fields:
        print("\nPlease provide the following missing information:")
        for field in null_fields:
            value = input(f"Enter {field}: ")
            json_data[field] = value
        print("\nUpdated event details:")
        print(json.dumps(json_data, indent=2))
    servicesList = get_service_recommendations(json_data)
    for i in range(0,len(servicesList)-1):
        print(f"service {i+1}")
        print(servicesList[i]['service_id'])
        print(servicesList[i]['category'])
        print(servicesList[i]['title'])
        print(servicesList[i]['description'])
        print("-"*50)
    print("="*50)
    print(get_package_recommendations(json_data))
    




    
