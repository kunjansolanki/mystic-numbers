import nltk
import string
import heapq

# --- NLTK Data Download ---
# This checks if the packages are available, and if not, downloads them.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab') # Added this package
except LookupError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    print("Download complete.")


def summarize_text(text, num_sentences=3):
    """
    Summarizes a given text using a Term Frequency (TF) approach.
    
    Args:
        text (str): The text to summarize.
        num_sentences (int): The number of sentences desired in the summary.
        
    Returns:
        str: The summarized text.
    """
    
    print("Starting summarization process...")

    # --- Step 1: Preprocessing & Word Frequency Calculation ---
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuation = string.punctuation
    words = nltk.word_tokenize(text.lower())
    
    word_frequencies = {}
    for word in words:
        if word not in stop_words and word not in punctuation:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    if not word_frequencies:
        return "Error: No valid words found in text."
        
    max_frequency = max(word_frequencies.values())

    # --- Step 2: Normalize Frequencies ---
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # --- Step 3: Score Sentences ---
    sentences = nltk.sent_tokenize(text)
    
    # Ensure we don't ask for more sentences than are available
    if num_sentences > len(sentences):
        print(f"Warning: Requested {num_sentences} sentences, but text only has {len(sentences)}. Returning full text.")
        return " ".join(sentences)
        
    sentence_scores = {}
    for sent in sentences:
        words_in_sentence = nltk.word_tokenize(sent.lower())
        for word in words_in_sentence:
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                    
    # --- Step 4: Select Top Sentences ---
    top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # --- Step 5: Assemble the Summary (in correct order) ---
    summary = ""
    for sent in sentences:
        if sent in top_sentences:
            summary += " " + sent
            
    return summary.strip()

# --- Main execution block ---
if __name__ == "__main__":
    
    # --- Get Text Input from User ---
    print("Please paste your text below. Press ENTER on an empty line when you are finished:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    user_text = "\n".join(lines)
    
    # --- Get Sentence Count from User ---
    while True:
        try:
            num_str = input("How many sentences do you want in the summary? (e.g., 3): ")
            num_sentences = int(num_str)
            if num_sentences <= 0:
                print("Please enter a positive number.")
                continue
            break # Exit loop if conversion is successful
        except ValueError:
            print("That's not a valid number. Please try again.")

    # Call the summarization function with the user's input
    summary = summarize_text(user_text, num_sentences=num_sentences)
    
    # Print the results
    print("\n--- GENERATED SUMMARY ---")
    print(summary)
