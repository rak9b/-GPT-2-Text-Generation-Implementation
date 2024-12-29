class TextGenerator:
    def __init__(self, model_path):
        """Initialize the generator with a trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)

    def generate(self, prompt, max_length=100, temperature=0.7, num_return_sequences=1):
        """Generate text based on a given prompt."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return generated_texts[0] if num_return_sequences == 1 else generated_texts
