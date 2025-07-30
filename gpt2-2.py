import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class BusinessImpactAnalyzer:
    def __init__(self, model_name="gpt2-large"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def analyze(self, input_text):
        try:
            # Determine input type based on length
            input_length = len(input_text.split())
            if input_length <= 10:  # Short Inputs
                params = {
                    "max_length": 80,
                    "min_length": 30,
                    "num_beams": 2,
                    "temperature": 0.7
                }
            elif 10 < input_length <= 40:  # Medium Inputs
                params = {
                    "max_length": 100,
                    "min_length": 50,
                    "num_beams": 4,
                    "temperature": 0.6
                }
            else:  # Long/Complex Inputs
                params = {
                    "max_length": 150,
                    "min_length": 70,
                    "num_beams": 6,
                    "temperature": 0.3
                }

            # Additional quality-based conditionals
            keywords = input_text.lower().split()
            if any(word in keywords for word in ["delay", "delays"]):
                params["temperature"] = 0.5  # Adjust temperature for delay-related inputs
            if any(word in keywords for word in ["cost", "expense", "expenses"]):
                params["num_beams"] = 5  # Increase beams for cost-related inputs
            if any(word in keywords for word in ["failure", "failures"]):
                params["max_length"] = 120  # Increase max length for failure-related inputs
            if any(word in keywords for word in ["market", "competition"]):
                params["temperature"] = 0.4  # Adjust temperature for market-related inputs

            # Generate prompt
            prompt = (
                f"Analyze the following business situation and provide the cause and effect:\n\n"
                f"Situation: {input_text}\n\n"
                "Cause and Effect Analysis:"
            )
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,  # Increased for more context
                truncation=True
            )

            outputs = self.model.generate(
                inputs,
                **params,
                no_repeat_ngram_size=3,  # Increased for better variety
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = generated_text.split("Cause and Effect Analysis:")[-1].strip()
            return self.format_analysis(analysis, input_text)

        except Exception as e:
            return f"Error in analysis: {str(e)}"

    def format_analysis(self, text, input_text):
        text = self.clean_text(text)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        cause_keywords = ['due to', 'driven by', 'caused by', 'result of', 'stems from', 
                         'because', 'factors', 'reasons', 'leads to', 'attributed to']
        effect_keywords = ['impact', 'result in', 'consequence', 'effect', 'cost', 
                         'expense', 'increase', 'decrease', 'affect', 'influence']
        
        cause_text = []
        effect_text = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in cause_keywords):
                cause_text.append(sentence)
            elif any(keyword in sentence.lower() for keyword in effect_keywords):
                effect_text.append(sentence)
            elif 'cost' in sentence.lower() or 'expense' in sentence.lower():
                effect_text.append(sentence)
            else:
                cause_text.append(sentence)

        if not cause_text:
            cause_text = ["The cause of the issue was not clearly identified."]
        if not effect_text:
            effect_text = ["The effect of the issue was not clearly identified."]

        # Take the first sentence for cause and effect
        cause = cause_text[0] if cause_text else "The cause of the issue was not clearly identified."
        effect = effect_text[0] if effect_text else "The effect of the issue was not clearly identified."

        return f"Cause: {cause}\n\nEffect: {effect}"

    def clean_text(self, text):
        cleanup_patterns = [
            r'in general,?',
            r'typically,?',
            r'it is important to note,?',
            r'there are several types of',
            r'in the case of',
            r'for example,?',
            r'such as',
            r'and so on',
            r'etc\.?'
        ]
        
        for pattern in cleanup_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = ' '.join(text.split())
        text = text.strip()
        
        if text:
            text = text[0].upper() + text[1:]
            if not text.endswith('.'):
                text += '.'
                
        return text

# Create Gradio interface
interface = gr.Interface(
    fn=lambda text: BusinessImpactAnalyzer().analyze(text),
    inputs=gr.Textbox(
        lines=4,
        label="Situation Description",
        placeholder="Describe the business situation or challenge..."
    ),
    outputs=gr.Textbox(label="Cause and Effect Analysis"),
    title="Business Impact Analyzer",
    description="Analyze business situations for their causes and impacts.",
    examples=[
        ["Additional costs due to expected fluctuation of staff"],
        ["Increased operational expenses from market volatility"],
        ["System downtime causing production delays and customer dissatisfaction"],
        ["New competitor entry affecting market share and pricing strategy"]
    ]
)

if __name__ == "__main__":
    interface.launch(share=True)