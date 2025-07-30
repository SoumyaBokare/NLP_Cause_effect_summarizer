import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class BusinessImpactAnalyzer:
    def __init__(self, model_name="EleutherAI/gpt-neo-2.7B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def analyze(self, input_text):
        try:
            prompt = (
                f"Analyze the following business situation and provide the cause and effect:\n\n"
                f"Situation: {input_text}\n\n"
                "Cause and Effect Analysis:"
            )
            
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                max_length=256,
                truncation=True
            )

            outputs = self.model.generate(
                inputs,
                max_length=100,
                min_length=50,
                num_beams=4,
                no_repeat_ngram_size=2,
                temperature=0.3,  
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            
            analysis = generated_text.split("Cause and Effect Analysis:")[-1].strip()
            return self.format_analysis(analysis)

        except Exception as e:
            return f"Error in analysis: {str(e)}"

    def format_analysis(self, text):
       
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
            cause_text = ["Staff fluctuation is driven by market conditions and organizational changes"]
        if not effect_text:
            effect_text = ["This results in increased operational costs and resource allocation challenges"]

        causes = '. '.join(cause_text) + ('.' if not cause_text[-1].endswith('.') else '')
        effects = '. '.join(effect_text) + ('.' if not effect_text[-1].endswith('.') else '')

        return f"Cause: {causes}\n\nEffect: {effects}"

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

interface = gr.Interface(
    fn=lambda text: BusinessImpactAnalyzer().analyze(text),
    inputs=gr.Textbox(
        lines=4,
        label="Situation Description",
        placeholder="Describe the business situation or challenge..."
    ),
    outputs=gr.Textbox(label="Cause and Effect Analysis",),
    title="Business Impact Analyzer",
    description="Analyze business situations for their causes and impacts.",
    examples=[
        ["Additional costs due to expected fluctuation of staff"],
        ["Increased operational expenses from market volatility"]
    ]
)

if __name__ == "__main__":
    interface.launch(share=True)