import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import joblib

# Load the trained sentiment classifier and the response generator
def load_models_and_tokenizers():
    # Load the saved emotion classification model and tokenizer
    emotion_model = GPT2ForSequenceClassification.from_pretrained("emotion_model")
    emotion_tokenizer = GPT2Tokenizer.from_pretrained("emotion_model")
    emotion_tokenizer.pad_token = emotion_tokenizer.eos_token

    # Load GPT-2 model and tokenizer for response generation
    response_model = GPT2LMHeadModel.from_pretrained("gpt2")
    response_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    response_tokenizer.pad_token = response_tokenizer.eos_token

    # Load sentence transformer model for similarity search
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    return emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model

def load_emotions():
    # Load saved emotions
    emotions = joblib.load("emotion_model/emotions.pkl")
    return emotions

def load_dataset():
    # Load the dataset used for training; assuming 'situation' and 'response' columns exist
    data = pd.read_csv("dataset.csv")
    situations = data['situation'].tolist()
    responses = data.get('dialogues', [""] * len(situations)).tolist()  # Assuming a 'response' column exists; use empty if not
    return situations, responses

def predict_emotion(emotion_model, emotion_tokenizer, text, device, emotions):
    max_length = 128
    inputs = emotion_tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    emotion_model.eval()
    with torch.no_grad():
        outputs = emotion_model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    confidence = probabilities[prediction].item()
    emotion = emotions[prediction]

    return emotion, confidence

def retrieve_similar_example(input_text, situations, responses, similarity_model):
    # Encode the input and all dataset examples using sentence transformer
    input_embedding = similarity_model.encode(input_text, convert_to_tensor=True)
    situation_embeddings = similarity_model.encode(situations, convert_to_tensor=True)

    # Compute similarity scores
    cosine_scores = util.pytorch_cos_sim(input_embedding, situation_embeddings)[0]

    # Find the most similar example
    best_match_idx = torch.argmax(cosine_scores).item()

    # Retrieve the most similar situation and its corresponding response
    similar_situation = situations[best_match_idx]
    response = responses[best_match_idx]
    return similar_situation, response, cosine_scores[best_match_idx].item()

def generate_response_from_example(similar_situation, response, response_model, response_tokenizer, user_input, device, emotion):
    if response:
        prompt = (f"User said: {user_input}\n"
                  f"Their emotion identified is: {emotion}\n"
                  f"It is a similar situation to: {similar_situation}\n"
                  f"Take reference of this: {response} to "
                  f"Generate a relevant and empathetic 2 sentence response.")
    else:
        prompt = (f"The user expressed feelings of {emotion}.\n"
                  f"Input: {user_input}\n"
                  f"Provide a comforting and empathetic response.")

    print(f"Prompt: {prompt}")

    inputs = response_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True).to(device)

    response_model.eval()
    with torch.no_grad():
        output = response_model.generate(
            inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            pad_token_id=response_tokenizer.eos_token_id
        )

    generated_response = response_tokenizer.decode(output[0], skip_special_tokens=True)
    generated_response = generated_response.split("Generate a relevant and empathetic 2 sentence response.")[-1].strip()

    # Fallback if response is empty or not meaningful
    if not generated_response.strip():
        generated_response = "I'm here to listen and help you through this. Can you tell me more about how you're feeling?"

    return generated_response



def main():
    # Load models, tokenizers, and dataset
    emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model = load_models_and_tokenizers()
    emotions = load_emotions()
    situations, responses = load_dataset()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emotion_model.to(device)
    response_model.to(device)

    print("\nNow you can test the model with your own inputs.")
    print("Type 'quit' to exit the program.")

    while True:
        user_input = input("\nEnter a text to analyze: ")
        if user_input.lower() == 'quit':
            break

        # Predict the emotion of the user input
        emotion, confidence = predict_emotion(emotion_model, emotion_tokenizer, user_input, device, emotions)

        # Retrieve a similar example and its response from the dataset
        similar_situation, retrieved_response, similarity_score = retrieve_similar_example(user_input, situations, responses, similarity_model)

        # Generate a refined response using the retrieved example
        response = generate_response_from_example(similar_situation, retrieved_response, response_model, response_tokenizer, user_input, device,emotion)

if _name_ == "_main_":
    main()