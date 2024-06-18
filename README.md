# EduGenQA-Intelligent-Q-A-System-for-E-learning-via-GoogleGenerativeAI-and-Langchain

EduGenQA aims to revolutionize the e-learning experience by leveraging advanced natural language processing capabilities provided by GoogleGenerativeAI and Langchain. This project focuses on developing a robust question-and-answer system tailored for an e-learning company with a diverse community of learners. Utilizing a structured knowledgebase derived from a CSV file, learners can interact through a user-friendly Streamlit interface to pose queries and receive accurate, timely responses. By harnessing the power of state-of-the-art language models, EduGenQA enhances accessibility and efficiency in educational support, transforming how learners engage and learn within digital environments.


![Screenshot 2024-06-18 104537](https://github.com/sushmamareddy/EduGenQA-Intelligent-Q-A-System-for-E-learning-via-GoogleGenerativeAI-and-Langchain/assets/36449873/6bc84b69-3580-4c3b-9fbd-9c50d52a9a58)

# Project Highlights
1. Use a real CSV file of FAQs that Codebasics company is using right now.
2. Their human staff will use this file to assist their course learners.
3. We will build an LLM based question and answer system that can reduce the workload of their human staff.
4. Students should be able to use this system to ask questions directly and get answers within seconds

# Project Structure
1. main.py: The main Streamlit application script.
2. langchain_helper.py: This has all the langchain code
3. requirements.txt: A list of required Python packages for the project


# Usage
1. Run the Streamlit app by executing:
   > streamlit run main.py
2. The web app will open in your browser.

    > To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.
    > Once knowledge base is created you will see a directory called faiss_index in your current folder
    > Now you are ready to ask questions. Type your question in Question box and hit Enter
