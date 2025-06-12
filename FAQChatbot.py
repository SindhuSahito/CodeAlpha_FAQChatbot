import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FAQChatbot:
    def __init__(self):
        # Initialize FAQ database (question: answer)
        self.faqs = {
            "What is your return policy?": "We accept returns within 30 days of purchase with original receipt.",
            "How long does shipping take?": "Standard shipping takes 3-5 business days. Express shipping is 1-2 days.",
            "Do you offer international shipping?": "Yes, we ship to most countries worldwide. Delivery times vary.",
            "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
            "How can I track my order?": "You'll receive a tracking number via email once your order ships.",
            "What are your customer service hours?": "Our customer service is available 9am-6pm EST, Monday to Friday.",
            "Do you have a physical store?": "We currently operate online only, with no physical retail locations.",
            "Can I cancel my order?": "Orders can be canceled within 1 hour of placement if they haven't been processed.",
            "What if my product is defective?": "Contact us within 14 days for defective products for a replacement or refund.",
            "Do you offer student discounts?": "Yes, we offer 10% off for students with valid .edu email addresses."
        }
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Preprocess all FAQ questions
        self.preprocessed_faqs = {self._preprocess_text(q): a for q, a in self.faqs.items()}
        self.faq_questions = list(self.preprocessed_faqs.keys())
        self.faq_answers = list(self.preprocessed_faqs.values())
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.faq_questions)
    
    def _preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_response(self, user_question):
        # Preprocess user question
        processed_question = self._preprocess_text(user_question)
        
        # Vectorize the question
        question_vector = self.vectorizer.transform([processed_question])
        
        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(question_vector, self.question_vectors)
        
        # Get index of most similar question
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[0, most_similar_idx]
        
        # Only return answer if similarity score is above threshold
        if similarity_score > 0.5:
            return self.faq_answers[most_similar_idx]
        else:
            return "I'm sorry, I don't have an answer for that question. Please contact customer support for further assistance."
    
    def run_chat(self):
        print("FAQ Chatbot: Hello! I can answer questions about our products and services. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("FAQ Chatbot: Goodbye!")
                break
                
            response = self.get_response(user_input)
            print(f"FAQ Chatbot: {response}")

# Create and run the chatbot
if __name__ == "__main__":
    chatbot = FAQChatbot()
    chatbot.run_chat()