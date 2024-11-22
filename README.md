This is the Neural Network Architecture
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define the columns to encode (assuming they are already encoded)
columns_to_encode = [
    'testphase', 'testexecutiontype', 'targetversion',
    'systemenvironment', 'storypoints', 'status', 'severity', 'rootcause',
    'issuetype', 'priority'
]

# Encode the target column ('Time_Spent_Category') if not already encoded
le_target = LabelEncoder()
df_object['Time_Spent_Category'] = le_target.fit_transform(df_object['Time_Spent_Category'])

# Initialize an empty list to store results
results = []

# Loop over each unique pkey
for pkey in df_object['pkey'].unique():
    print(f"Processing pkey: {pkey}")

    # Filter the data for the current pkey
    df_pkey = df_object[df_object['pkey'] == pkey]

    # Separate features and target for the current pkey
    X = df_pkey[columns_to_encode]
    y = df_pkey['Time_Spent_Category']

    # Ensure no missing values
    if X.isnull().sum().any() or y.isnull().sum():
        raise ValueError("Missing values found in features or target.")

    # Convert target to categorical (for classification)
    y = to_categorical(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    NotBOT = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')  # Adjust activation if binary or multiclass
    ])

    # Compile the model
    NotBOT.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    NotBOT.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = NotBOT.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Make predictions and store classification report
    y_pred = NotBOT.predict(X_test)
    y_pred_class = y_pred.argmax(axis=1)
    y_test_class = y_test.argmax(axis=1)

    report = classification_report(y_test_class, y_pred_class)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)  # Normalize rows

    print(f"Confusion Matrix for pkey {pkey}:")
    print(conf_matrix)

    # Append results
    results.append({
        'pkey': pkey,
        'loss': loss,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'normalized_confusion_matrix': conf_matrix_normalized
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display the results for each pkey
print(results_df)

# Decode back target labels for human readability (if needed)
print("\nTarget Label Mapping:")
print(dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))
