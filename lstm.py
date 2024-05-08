
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Load data
df = pd.read_csv('/Users/Agaaz/Downloads/raw_pe_images.csv')

# Exclude non-feature columns and print remaining features
features = df.columns.drop(['hash'])
print("Features included in PCA:", features)

# Encode the 'legitimate' column
le = LabelEncoder()
df['malware'] = le.fit_transform(df['malware'])

# Extract features and standardize them
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

# Apply PCA without limiting the number of components to inspect the variance
pca = PCA(n_components=None)
principalComponents = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Determine the number of components needed to explain at least 95% of the variance
n_components_95 = (cumulative_variance < 0.95).sum() + 1
print(f"Number of components to explain 95% of variance: {n_components_95}")

# Reapply PCA with the determined number of components
pca = PCA(n_components=n_components_95)
principalComponents = pca.fit_transform(x)

# Split data into features and labels
X = principalComponents
y = df['malware']
#this is final data post pCA analysis with 36 features.
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)



model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(LSTM(32))  # Add LSTM layer; 32 units is an arbitrary choice, adjust as needed
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}')
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()