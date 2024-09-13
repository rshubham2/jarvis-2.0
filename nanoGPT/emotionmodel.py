import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib  # For saving the LabelEncoder

class EmotionDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }

def load_data():
    data = pd.read_csv("emotion_dataset.csv")
    texts = data['Clean_Text'].tolist()[:500]
    labels = data['Emotion'].tolist()[:500]

    # Use LabelEncoder to convert string labels to numeric
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    # Get the list of unique emotions
    emotions = label_encoder.classes_.tolist()

    return train_test_split(texts, numeric_labels, test_size=0.2, random_state=42), emotions, label_encoder

def train_model(model, train_loader, val_loader, device, epochs=5):
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_true, val_preds)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

def main():
    # Load and preprocess data
    (train_texts, val_texts, train_labels, val_labels), emotions, label_encoder = load_data()

    print(f"Emotions in the dataset: {emotions}")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(emotions))
    model.config.pad_token_id = model.config.eos_token_id

    # Prepare datasets and dataloaders
    max_length = 128
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    train_model(model, train_loader, val_loader, device)

    # Save the model, tokenizer, emotions, and label encoder
    model.save_pretrained("emotion_model")
    tokenizer.save_pretrained("emotion_model")
    joblib.dump(emotions, "emotion_model/emotions.pkl")
    joblib.dump(label_encoder, "emotion_model/label_encoder.pkl")
    print("Model, tokenizer, emotions, and label encoder saved successfully.")

if _name_ == "_main_":
    main()