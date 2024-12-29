def preprocess_text(text):
    """Preprocess text by cleaning and adding end-of-text token."""
    text = ' '.join(text.split())  # Remove extra whitespace
    text += ' <|endoftext|>'  # Add GPT-2's end-of-text token
    return text

def load_text_file(file_path):
    """Load text from a file and preprocess it."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return preprocess_text(text)
