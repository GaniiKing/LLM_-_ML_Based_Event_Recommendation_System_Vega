import random
import joblib
import pandas as pd
import numpy as np
import ast

# Load all components from a single file
model_components = joblib.load('service_recommender.pkl')
clf = model_components['model']
encoders = model_components['encoders']
tfidf_instructions = model_components['tfidf_instructions']
mlb = model_components['mlb']

# Input data for new booking - only essential fields
input_data = {
  "event": "Wedding",
  "vibe": "Youthful",
  "date": "2025-08-07",
  "budget": 70000,
  "time": "20:00",
  "duration": 9,
  "venue": "Indoor",
  "location": "Hyderabad",
  "guests": 500,
  "special_instructions": "Front seats reserved for elderly people"
}

def get_event_recommendations():
    events = pd.read_csv('vega_detailed_csv_files\generated_events.csv')
    events_i = events[events['event_type']==input_data['event']]
    
    if len(events_i) == 0:
        print(f"\nNo events found for event type: {input_data['event']}")
        print("Available event types:", events['event_type'].unique())
        return None, None
    
    # Calculate budget flexibility (10% higher)
    flexible_budget = input_data['budget'] * 1.1
    
    # Filter events within budget constraints
    budget_filtered_events = events_i[events_i['cost'] <= flexible_budget]
    
    if len(budget_filtered_events) == 0:
        print(f"\nNo events found within budget constraints (₹{input_data['budget']:,} - ₹{flexible_budget:,.0f})")
        return None, None
    
    # Sort by cost to get the best match within budget
    budget_filtered_events = budget_filtered_events.sort_values('cost', ascending=False)
    
    # Get the highest cost event that's still within budget
    selected_event = budget_filtered_events.iloc[0]
    
    # Parse the services_included string into a list of dictionaries
    selected_event['services_included'] = ast.literal_eval(selected_event['services_included'])
    
    # Check if we're using the flexible budget
    if selected_event['cost'] > input_data['budget']:
        print(f"\nNote: Selected event slightly exceeds original budget by ₹{selected_event['cost'] - input_data['budget']:,.0f}")
        print(f"Using flexible budget of ₹{flexible_budget:,.0f}")
    
    return selected_event['package_id'], selected_event

def get_service_recommendations(input_data, print_details=True):
    # Preprocess input data
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features using the loaded encoders
    input_df['event_id'] = encoders['event_id'].transform(input_df['event'])
    input_df['vibe'] = encoders['vibe'].transform(input_df['vibe'])
    input_df['location'] = encoders['location'].transform(input_df['location'])
    input_df['venue'] = encoders['venue'].transform(input_df['venue'])
    
    # Process special instructions
    instructions_tfidf_features = tfidf_instructions.transform(input_df['special_instructions'].astype(str))
    
    # Combine features including duration and guests
    X_input = np.hstack([
        input_df[['event_id', 'vibe', 'location', 'budget', 'venue', 'duration', 'guests']].values,
        instructions_tfidf_features.toarray()
    ])
    
    # Make predictions
    predictions = clf.predict(X_input)
    
    # Get prediction probabilities
    prediction_probs = clf.predict_proba(X_input)
    
    # Get confidence scores for each service
    confidence_scores = []
    for i, service in enumerate(mlb.classes_):
        # Skip if service is "No" or invalid
        if service.lower() in ['no', 'n', 'o'] or len(service) <= 1:
            continue
            
        # Get probability of positive class from the correct array structure
        service_probs = prediction_probs[i]
        if isinstance(service_probs, np.ndarray) and len(service_probs.shape) > 1:
            confidence = service_probs[0][1]  # Get probability of positive class
        else:
            confidence = service_probs[1] if len(service_probs) > 1 else service_probs[0]
        
        # Include all services with any positive confidence
        if confidence > 0.1:  # Very low threshold to get more recommendations
            confidence_scores.append((service, confidence))
    
    # Sort services by confidence score
    sorted_services = sorted(confidence_scores, key=lambda x: x[1], reverse=True)
    
    # Take top 5 services or all if less than 5
    top_services = sorted_services[:5] if len(sorted_services) > 5 else sorted_services
    
    if print_details:
        print("\nInput Details:")
        print(f"Event: {input_data['event']}")
        print(f"Vibe: {input_data['vibe']}")
        print(f"Location: {input_data['location']}")
        print(f"Budget: {input_data['budget']}")
        print(f"Duration: {input_data['duration']} hours")
        print(f"Venue: {input_data['venue']}")
        print(f"Guests: {input_data['guests']}")
        print(f"Special Instructions: {input_data['special_instructions']}")
        
        all_services = pd.read_csv('vega_detailed_csv_files\generated_providers.csv')
        
        if not top_services:
            print("\nNo services were recommended. This might be due to:")
            print("1. No matching services in the training data")
            print("2. Low confidence scores for all services")
            print("3. Input values not matching the training data categories")
        else:
            print("\nRecommended service package (sorted by confidence):")
            for service, confidence in top_services:
                jammed = all_services[all_services['service_id'] == service]
                print(f"--{service} (Confidence: {confidence:.2%})")
                print(f"      {jammed['service_category'].values[0]}")
                print(f"      {jammed['service_title'].values[0]}")
                print(f"      {jammed['service_description'].values[0]}")
                print("------------------------------------------------")
    
    return top_services

# Example usage:
# First scenario with original budget
print("\n=== SCENARIO 1: Original Budget ===")
print(f"Budget: ₹{input_data['budget']:,}")
get_service_recommendations(input_data)

# Second scenario with increased budget
print("\n=== SCENARIO 2: Increased Budget (+10%) ===")
input_data['budget'] = int(input_data['budget'] * 1.1)  # Increase budget by 10%
print(f"Budget: ₹{input_data['budget']:,}")
get_service_recommendations(input_data)

print("=== OPTIONAL: Get event recommendations ===")
event_id,event_data = get_event_recommendations()
print("--",event_id)
print(f"Event Type: {event_data['event_type']}")
print(f"Event Budget: {event_data['cost']}")
length = len(event_data['services_included'])
for i in range(length):
    print(f"Service {i+1}: {event_data['services_included'][i]['title']}")
    print(f"     {event_data['services_included'][i]['category']}")
    print(f"     {event_data['services_included'][i]['description']}")





