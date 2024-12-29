

def preprocess_prompt(prompt):
    """Preprocess the prompt by removing unnecessary whitespace and capitalization."""
    return ' '.join(prompt.split()).capitalize()


def validate_output_directory(output_dir):
    """Ensure the output directory exists and is writable."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory {output_dir} is not writable.")


def load_prompts_from_file(file_path):
    """Load prompts from a text file, each line representing a separate prompt."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts
