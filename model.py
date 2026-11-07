import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset.csv")

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(df["feedback"]).toarray()
text_df = pd.DataFrame(text_features)

# Encode categorical columns
cols = [
"class_time",
"day",
"subject_difficulty",
"teaching_style",
"weather",
"student_interest",
"exam_near"
]

encoder = LabelEncoder()

for col in cols:
    df[col] = encoder.fit_transform(df[col])

# Combine features
X = pd.concat([df.drop(["attendance","feedback"], axis=1), text_df], axis=1)

# Convert column names to string
X.columns = X.columns.astype(str)

# Encode target
attendance_encoder = LabelEncoder()
y = attendance_encoder.fit_transform(df["attendance"])

# Train model
model = RandomForestClassifier()
model.fit(X, y)


def predict_attendance(data, feedback):

    # Convert feedback to vector
    text_vec = vectorizer.transform([feedback]).toarray()
    text_df = pd.DataFrame(text_vec)

    # Convert input data to dataframe
    data_df = pd.DataFrame([data])

    # Combine input + text
    final_input = pd.concat([data_df, text_df], axis=1)

    # Convert column names to string
    final_input.columns = final_input.columns.astype(str)

    # Predict
    pred = model.predict(final_input)

    # Return original label
    return attendance_encoder.inverse_transform(pred)[0]