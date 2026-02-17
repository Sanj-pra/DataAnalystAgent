"""
Simple Data Analysis Agent using Azure OpenAI
A beginner-friendly agentic AI project

This agent can:
1. Read CSV data files
2. Analyze the data using AI
3. Answer questions about your data
4. Suggest insights and visualizations
"""

import os
from openai import AzureOpenAI
import pandas as pd
import json

class DataAnalysisAgent:
    def __init__(self, api_key, endpoint, deployment_name):
        """
        Initialize the agent with Azure OpenAI credentials
        
        Args:
            api_key: Your Azure OpenAI API key
            endpoint: Your Azure OpenAI endpoint URL
            deployment_name: Your deployment name (e.g., 'gpt-4')
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        self.conversation_history = []
        
    def load_data(self, file_path):
        """Load a CSV file and return basic info about it"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully!")
            print(f"  - Rows: {len(self.data)}")
            print(f"  - Columns: {len(self.data.columns)}")
            print(f"  - Column names: {', '.join(self.data.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def get_data_summary(self):
        """Create a text summary of the data for the AI"""
        if not hasattr(self, 'data'):
            return "No data loaded yet."
        
        summary = f"""
Data Overview:
- Total rows: {len(self.data)}
- Columns: {', '.join(self.data.columns)}

First few rows:
{self.data.head(10).to_string()}

Basic statistics:
{self.data.describe().to_string()}
"""
        return summary
    
    def analyze(self, question):
        """
        Send a question to the AI agent about your data
        
        The agent will think step-by-step and provide insights
        """
        if not hasattr(self, 'data'):
            return "Please load data first using load_data()"
        
        # Prepare the prompt with data context
        system_message = """You are a helpful data analysis assistant. 
You help users understand their data by:
1. Analyzing patterns and trends
2. Answering specific questions
3. Suggesting useful visualizations
4. Explaining insights in simple terms

When analyzing data, think step-by-step and be specific."""

        user_message = f"""Here is the data I'm working with:

{self.get_data_summary()}

Question: {question}

Please analyze this and provide clear, actionable insights."""

        # Call Azure OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer
            })
            
            return answer
            
        except Exception as e:
            return f"Error calling Azure OpenAI: {e}"
    
    def suggest_next_steps(self):
        """Ask the AI what analysis would be most valuable"""
        return self.analyze(
            "Based on this data, what are the 3 most interesting "
            "questions I should explore? What insights might be hidden here?"
        )
    
    def generate_visualization_code(self, chart_description):
        """
        Ask the AI to generate Python code for a specific visualization
        
        Example: "Create a bar chart showing sales by region"
        """
        prompt = f"""Given this data structure:
{self.get_data_summary()}

Generate Python code using matplotlib or seaborn to create this visualization:
{chart_description}

Provide only the code, starting with necessary imports. Assume the data is already loaded in a variable called 'data'."""

        response = self.analyze(prompt)
        return response


# Example usage function
def example_usage():
    """
    Example of how to use the Data Analysis Agent
    Replace with your own Azure OpenAI credentials
    """
    
    # ========== CONFIGURATION ==========
    # Replace these with your actual Azure OpenAI details
    API_KEY = "your-api-key-here"
    ENDPOINT = "https://your-resource-name.openai.azure.com/"
    DEPLOYMENT_NAME = "your-deployment-name"  # e.g., "gpt-4"
    
    # Initialize the agent
    agent = DataAnalysisAgent(
        api_key=API_KEY,
        endpoint=ENDPOINT,
        deployment_name=DEPLOYMENT_NAME
    )
    
    # Load your data
    # Replace 'your_data.csv' with your actual CSV file
    agent.load_data('sample_sales_data.csv')
    
    # Ask questions about your data
    print("\n" + "="*50)
    print("ANALYSIS 1: Overall Summary")
    print("="*50)
    result = agent.analyze("What are the main trends in this data?")
    print(result)
    
    print("\n" + "="*50)
    print("ANALYSIS 2: Get AI Suggestions")
    print("="*50)
    suggestions = agent.suggest_next_steps()
    print(suggestions)
    
    # You can keep asking questions
    # result = agent.analyze("What is the average value by category?")
    # print(result)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════╗
    ║   Data Analysis Agent - Azure OpenAI Edition   ║
    ║                                                ║
    ║   Your agentic AI project!               ║
    ╚════════════════════════════════════════════════╝
    
    Before running, update the configuration in example_usage():
    1. Add your Azure OpenAI API key
    2. Add your Azure OpenAI endpoint
    3. Add your deployment name
    4. Prepare a CSV file to analyze
    
    Then uncomment the line below to run the example.
    """)
    
    # Uncomment this line once you've configured everything:
    # example_usage()
