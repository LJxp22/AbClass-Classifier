import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns


# import shap  # Uncomment for feature importance analysis

def load_data(file_path):
    """Loads and preprocesses the antibody classification dataset"""
    database = pd.read_csv(file_path)

    # Convert non-label columns to numeric
    numeric_columns = database.columns[1:]
    database[numeric_columns] = database[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Remove rows with missing values
    database.dropna(inplace=True)

    # Convert label to float32
    database['label'] = database['label'].astype('float32')

    return database

def prepare_data(database):
    X = database.drop('label', axis=1)
    y = database['label']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

# Self-Attention layer implementation
class SelfAttention(keras.layers.Layer):
    """Self-Attention mechanism for the Transformer model"""

    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dense_q = keras.layers.Dense(embed_dim)
        self.dense_k = keras.layers.Dense(embed_dim)
        self.dense_v = keras.layers.Dense(embed_dim)
        self.dense_out = keras.layers.Dense(embed_dim)

    def call(self, inputs):
        query = self.dense_q(inputs)
        key = self.dense_k(inputs)
        value = self.dense_v(inputs)

        # Compute attention weights
        attention_weights = tf.nn.softmax(
            tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        )
        output = tf.matmul(attention_weights, value)
        output = self.dense_out(output)

        return output


# Feedforward network for Transformer encoder
def feed_forward_network(embed_dim, feedforward_dim):
    """Creates a feedforward neural network for the Transformer"""
    return keras.Sequential([
        keras.layers.Dense(feedforward_dim, activation='relu'),
        keras.layers.Dense(embed_dim)
    ])


# Transformer encoder layer
class EncoderLayer(keras.layers.Layer):
    """Single encoder layer of the Transformer model"""

    def __init__(self, embed_dim, feedforward_dim, num_heads, dropout_rate=0.1, l2_regularization=0.01):
        super(EncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.feed_forward = feed_forward_network(embed_dim, feedforward_dim)
        self.layernorm1 = keras.layers.LayerNormalization()
        self.layernorm2 = keras.layers.LayerNormalization()
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.regularization = keras.regularizers.l2(l2_regularization)

    def call(self, inputs, training=True):
        # Self-attention sublayer with residual connection and layer norm
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.layernorm1(inputs + attention_output)

        # Feedforward sublayer with residual connection and layer norm
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        output = self.layernorm2(attention_output + feed_forward_output)

        return output


# Transformer model
class Transformer(keras.Model):
    """Full Transformer model with multiple encoder layers"""

    def __init__(self, num_layers, embed_dim, feedforward_dim, num_heads, dropout_rate=0.1, l2_regularization=0.01):
        super(Transformer, self).__init__()
        self.encoder_layers = [
            EncoderLayer(embed_dim, feedforward_dim, num_heads, dropout_rate, l2_regularization)
            for _ in range(num_layers)
        ]

    def call(self, inputs, training=True):
        output = inputs
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output, training=training)
        return output


def model():
    # Model configuration parameters
    num_layers = 6
    embed_dim = 81
    feedforward_dim = 256
    num_heads = 8
    num_classes = 6
    l2_regularization = 0.56
    learning_rate = 0.001


    # Create Transformer model
    transformer = Transformer(
        num_layers=num_layers,
        embed_dim=embed_dim,
        feedforward_dim=feedforward_dim,
        num_heads=num_heads,
        l2_regularization=l2_regularization
    )

    # Define model architecture
    inputs = tf.keras.Input(shape=(X_train.shape[1],))
    x = transformer(inputs)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(l2_regularization)
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def evaluate_model(model,X_train, X_test, y_train, y_test):
    num_epochs = 256
    batch_size = 128
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    # Print accuracy for each epoch
    print("\nEpoch-wise Accuracy:")
    for epoch in range(num_epochs):
        train_acc = history.history['accuracy'][epoch]
        val_acc = history.history['val_accuracy'][epoch]
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Visualize training history
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Class names for visualization and reporting
    class_names = ['anti-dengue', 'anti-influenza', 'anti-tetanus', 'anti-sars-cov2', 'anti-Tuberculosis', 'other']

    # Calculate class-wise accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print("\nClass-wise Accuracy:")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {class_accuracy[i]:.4f}")

    # Overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Classification report
    class_report = classification_report(y_test, y_pred_classes, target_names=class_names)
    print("\nClassification Report:")
    print(class_report)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # # Uncomment for SHAP feature importance analysis
    # try:
    #     explainer = shap.Explainer(model, X_train)
    #     shap_values = explainer.shap_values(X_test)
    #     shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    #     plt.savefig('shap_summary.png')
    # except Exception as e:
    #     print(f"SHAP analysis not performed: {e}")
    #     print("Install 'shap' library for feature importance visualization.")

if __name__ == "__main__":
    # Load and preprocess data
    file_path = "data.csv"
    database = load_data(file_path)

    X_train, X_test, y_train, y_test = prepare_data(database)

    model = model()

    evaluate_model(model, X_train, X_test, y_train, y_test)













