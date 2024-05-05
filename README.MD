# Blog Generator 🤖

## Overview
This Streamlit application allows users to generate blogs using the LLAMA 2 language model. Users can input a topic for discussion, specify the number of words for the blog, and select the type of audience for the blog. The application then utilizes the LLAMA 2 model to generate a blog post based on the provided inputs.

## Features
- Input topic for discussion
- Specify the number of words for the blog
- Select the type of audience for the blog (e.g., Data Science guys, Research people, General Public)
- Generate a blog post using the LLAMA 2 model
- User-friendly interface with Streamlit

## Usage
1. Clone the repository:
git clone https://github.com/pandeyaditya1996/blog-generator.git
cd blog-generator

markdown
Copy code

2. Install dependencies:
pip install -r requirements.txt

markdown
Copy code

3. Run the Streamlit app:
streamlit run app.py

markdown
Copy code

4. Access the application in your browser at `http://localhost:8501`.

## Dependencies
- Streamlit
- langchain (assuming it's a package containing the `PromptTemplate` and `CTransformers` modules)

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
