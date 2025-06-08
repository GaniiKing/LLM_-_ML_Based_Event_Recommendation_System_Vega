import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import joblib

# Load the data
csv_path = 'finalbookings/generated_bookings.csv'
df = pd.read_csv(csv_path)

print("Initial data shape:", df.shape)

# Preprocessing
# Fill missing values
for col in ['event_id', 'vibe', 'location', 'budget', 'rating', 'reviews', 'special_instructions', 'venue', 'duration', 'guests']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Convert service_id to list (multi-label)
def parse_services(x):
    if pd.isna(x) or str(x).strip().lower() in ['no', '', 'n', 'o']:
        return []
    
    # Split by comma and clean each service ID
    services = str(x).split(',')
    # Clean each service ID
    cleaned_services = []
    for service in services:
        service = service.strip()
        # Only keep services that are not empty, not 'no', and have more than 1 character
        if (service and 
            service.lower() not in ['no', 'n', 'o'] and 
            len(service) > 1 and 
            not service.isdigit()):  # Exclude pure numbers
            cleaned_services.append(service)
    return cleaned_services

# Apply service parsing
df['service_id_list'] = df['service_id'].apply(parse_services)

# Print unique services for debugging
print("\nUnique services before cleaning:")
print(df['service_id'].unique())

print("\nUnique services after cleaning:")
print(set([service for services in df['service_id_list'] for service in services]))

# Remove rows with no services
df = df[df['service_id_list'].apply(len) > 0]
print("\nData shape after removing rows with no services:", df.shape)

# Calculate sentiment scores for reviews (using historical data)
def get_sentiment_score(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

df['review_sentiment'] = df['reviews'].apply(get_sentiment_score)

# Create a weighted score combining ratings and sentiment
df['service_score'] = (df['rating'].astype(float) * 0.7) + (df['review_sentiment'] * 0.3)

# Select features for recommendation
features = ['event_id', 'vibe', 'location', 'budget', 'special_instructions', 'venue', 'duration', 'guests']
X = df[features].copy()

# Create and fit encoders
encoders = {}
for col in ['event_id', 'vibe', 'location', 'venue']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Process special instructions
tfidf_instructions = TfidfVectorizer(max_features=100, stop_words='english')
instructions_tfidf = tfidf_instructions.fit_transform(X['special_instructions'].astype(str))

# Convert TF-IDF matrices to dense arrays
instructions_dense = instructions_tfidf.toarray()

# Combine features
X_combined = np.hstack([
    X[['event_id', 'vibe', 'location', 'budget', 'venue', 'duration', 'guests']].values,
    instructions_dense
])

# Multi-label binarizer for service_id
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['service_id_list'])

# Print unique services after binarization
print("\nUnique services after binarization:")
print(mlb.classes_)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

# Model: Multi-label Random Forest with class weights
class_weights = {i: 1.5 for i in range(Y.shape[1])}  # Give higher weight to positive classes
clf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weights
    )
)
clf.fit(X_train, Y_train)

# Create a dictionary with all components
model_components = {
    'model': clf,
    'encoders': encoders,
    'tfidf_instructions': tfidf_instructions,
    'mlb': mlb
}

# Save all components in a single file
joblib.dump(model_components, 'service_recommender.pkl')

print('\nAll model components saved in service_recommender.pkl') 