import streamlit as st
import json
from typing import List
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import requests
import google.generativeai as genai
from langchain_openai import AzureOpenAIEmbeddings
import base64
import streamlit_antd_components as sac

# Load environment variables
load_dotenv()

# Initialize services with direct values
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_internquest = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

# Initialize Pinecone with direct values
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pc.Index("lamjed")
# Define your page functions
with open('style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
 # Display HTML content
st.markdown("""
    <div class="area" >
        <ul class="circles">
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
        </ul>
                
    </div >
    """, unsafe_allow_html=True)
def home_page():
    st.session_state['b64_image'] =""
    with open("./pic.png", "rb") as img_file:
        img_back = base64.b64encode(img_file.read()).decode("utf-8")
        # st.image(f'data:image/png;base64,{img_back}', use_column_width=False)
        st.markdown(f"""<img class="back_img"  src="data:image/png;base64,{img_back}" alt="Frozen Image">""",unsafe_allow_html=True)
    st.markdown("""<h1 class="Title">Welcome To Tutorini</h1>""",unsafe_allow_html=True)

def setup_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        model="text-embedding-3-small",
        api_version="2024-02-01",
    )

def get_stem_question(topic: str):
    topic_prompts = {
        "PGBI-IKM": """Generate a question related to Process Governance and Business Intelligence - Information and Knowledge Management. 
        Focus on topics like data governance, information lifecycle, knowledge management systems, and business intelligence processes.
        The output should be in JSON format:
        {
            "question": "A detailed PGBI-IKM question",
            "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
            "correct_answer": "The correct choice",
            "explanation": "Detailed explanation of the answer"
        }
        """,
        "Digital Leadership": """Generate a question related to Digital Leadership and transformation.
        Focus on topics like digital strategy, organizational change, digital innovation, and leadership in the digital age.
        The output should be in JSON format:
        {
            "question": "A detailed Digital Leadership question",
            "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
            "correct_answer": "The correct choice",
            "explanation": "Detailed explanation of the answer"
        }
        """
    }
    
    prompt = topic_prompts.get(topic, topic_prompts["PGBI-IKM"])  # Default to PGBI-IKM if topic not found
    response = model_internquest.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.5)
    )
    return json.loads(response.text)


def name_space_picker(user_query: str) -> str:
    prompt = f"""
    You are a name_sapce picker for pinecone vecot data base the index contains these two name spaces ["pgbi-digital-leadership", "pgbi-ikm"] pick the correct one based on the user query : {user_query} . Note that the output should be in JSON format:
    {{
        "Name_Space": str 
    }}    
    """
    response = model_internquest.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0)
    )
    json_response = json.loads(response.text)
    return json_response["Name_Space"]

def generate_response(user_query: str,is_the_user_input_question : bool = False) -> str:
    """
    Generates a response by querying a vector database and using an AI model to construct the answer.
    
    Args:
        user_query (str): The query provided by the user.
    
    Returns:
        str: The correct answer generated from the database and/or AI model.
    """
    if is_the_user_input_question:
        # Step 1: Generate the query embedding using Azure
        embeddings = setup_embeddings()
        vector = embeddings.embed_query(user_query)
        # print("Generated Embedding Vector:", vector)  # Debugging print statement
        namespace = name_space_picker(user_query)
        # Step 2: Query the Pinecone index
        search_results = index.query(
            vector=vector,
            top_k=4,  # Retrieve the top 4 most relevant matches
            include_values=False,
            include_metadata=True,
            namespace=namespace
        )
        print(search_results)
        # Step 3: Handle search results and construct response
        if search_results and "matches" in search_results and search_results["matches"]:
            # Gather references from the metadata of matches
            references = [
                match["metadata"] for match in search_results["matches"]
            ]
            print("Search Results Metadata:", references)  # Debugging print statement

            # Create a detailed prompt using the references
            references_text = "\n".join(
                f"- {ref}" for ref in references
            )
            print(f" references_text : {references_text}")
            prompt = f"""
            Use the following references to answer the question:\n{references_text}\n
            Question: {user_query}
            """
            response = model_internquest.generate_content(prompt)
            correct_answer = response.text.strip()
        else:
            # No relevant results; fall back to direct AI answer generation
            print("Nothing is retreived !")
            prompt = f"Provide a detailed answer to the following question:\n{user_query}"
            response = model_internquest.generate_content(prompt)
            correct_answer = response.text.strip()
    else :
        prompt = f"Provide a detailed answer to the following question:\n{user_query}"
        response = model_internquest.generate_content(prompt)
        correct_answer = response.text.strip()
    print(correct_answer)
    return correct_answer
def verify_is_the_user_input_question(user_input: str) -> bool:
    prompt = f"""
    You are a user input verifier verify if the user input "{user_input}" is a question that is usual or not. Note that the output should be in JSON format:
    {{
        "user_input_is_usual_question": bool 
    }}    
    """
    response = model_internquest.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0)
    )
    json_response = json.loads(response.text)
    # print(f"Is it a question : {json_response["user_input_is_usual_question"]}")
    return json_response["user_input_is_usual_question"]


def verify_response(response: str, correct_answer: str) -> bool:
    prompt = f"""
    You are a response verifier verify if the user reponse "{response}" is almost the same as the correct answer "{correct_answer}" or not. Note that the output should be in JSON format:
    {{
        "Is_Correct": bool 
    }}    
    """
    response = model_internquest.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0)
    )
    json_response = json.loads(response.text)
    return json_response["Is_Correct"]

def generate_hint_questions_from_db(user_query: str, num_hints: int) -> List[str]:
    """
    Retrieve hints from the vector database (Pinecone) based on the user's query.
    Progressively make the hints more specific by using the top results from the search.
    """
    embeddings = setup_embeddings()
    vector = embeddings.embed_query(user_query)
    
    search_results = index.query(
        vector=vector,
        top_k=num_hints,
        include_values=False,
        include_metadata=True
    )
    
    if search_results and "matches" in search_results and search_results["matches"]:
        hints = [match["metadata"]["text"] for match in search_results["matches"]]
        return hints
    else:
        return ["Sorry, no hints available."]
def generate_hint_questions(number_of_hints: int, correct_answer: str) -> List[str]:
    prompt = f"""
    You are a Socratic question generator. Create {number_of_hints} progressively more detailed hints to guide the user toward discovering the correct answer "{correct_answer}".
    Start with broad conceptual questions and gradually become more specific.
    The hints should build upon each other, with each hint revealing slightly more information.
    The output should be in JSON format:
    {{
        "Questions": [
            "Hint 1: ...",
            "Hint 2: ...",
            ...
        ]
    }}
    """
    response = model_internquest.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0)
    )
    json_response = json.loads(response.text)
    return json_response.get("Questions", [])
 
def show_learning_assistant():
    # Initialize necessary session state variables
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'attempts' not in st.session_state:
        st.session_state.attempts = 0
    if 'max_attempts' not in st.session_state:
        st.session_state.max_attempts = 1
    if 'current_hint_index' not in st.session_state:
        st.session_state.current_hint_index = 0
    if 'hints' not in st.session_state:
        st.session_state.hints = []
    if 'correct_answer' not in st.session_state:
        st.session_state.correct_answer = None

    # Progress tracking
    progress_col1, progress_col2 = st.columns(2)
    with progress_col1:
        st.markdown("### Progress")
        progress_bar = st.progress(0)
    with progress_col2:
        st.markdown(f"### Attempts: {st.session_state.attempts}/{st.session_state.max_attempts}")

    if st.session_state.step == 0:
        st.subheader("💭 What would you like to learn about?")
        user_question = st.text_input("Enter your question:")
        
        if user_question:
            is_the_user_input_question = verify_is_the_user_input_question(user_question)
            with st.spinner("🤔 Preparing your learning journey..."):
                st.session_state.correct_answer = generate_response(user_question,is_the_user_input_question)
                if is_the_user_input_question:
                    st.session_state.hints = generate_hint_questions(3, st.session_state.correct_answer)
                    st.session_state.step = 1
                    st.rerun()
                else :
                    st.markdown(st.session_state.correct_answer)
                
    elif st.session_state.step == 1:
        st.subheader("🤔 Let's think about this together")
        
        if st.session_state.current_hint_index < len(st.session_state.hints):
            st.info(st.session_state.hints[st.session_state.current_hint_index])
        
        user_answer = st.text_area("What do you think? Use the hint above to guide your answer:", height=150)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Submit Answer", use_container_width=True):
                with st.spinner("🔍 Checking your answer..."):
                    is_correct = verify_response(user_answer, st.session_state.correct_answer)
                    
                    if is_correct:
                        st.success("🎉 Excellent! You've got it right!")
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.session_state.attempts += 1
                        if st.session_state.attempts >= st.session_state.max_attempts:
                            if st.session_state.current_hint_index < len(st.session_state.hints) - 1:
                                st.session_state.current_hint_index += 1
                                st.session_state.attempts = 0                                
                            else:                                  
                                st.session_state.step = 2
                                st.rerun()
                        else:
                            st.error(f"Not quite right. Try again! ({st.session_state.max_attempts - st.session_state.attempts} attempts remaining)")
        
        progress = (st.session_state.current_hint_index / len(st.session_state.hints))
        progress_bar.progress(progress)
    
    elif st.session_state.step == 2:
        st.subheader("📚 Learning Complete!")
        st.success("Here's the complete answer:")
        st.write(st.session_state.correct_answer)
        
        st.markdown("### 🤔 Reflection")
        st.write("Let's review what we learned:")
        for i, hint in enumerate(st.session_state.hints, 1):
            st.markdown(f"Step {i}: {hint}")
        
        if st.button("Start New Question"):
            for key in ['step', 'attempts', 'current_hint_index', 'hints', 'correct_answer']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 0
            st.rerun()


def get_top_5_answers(user_query: str, namespace: str) -> List[str]:
    embeddings = setup_embeddings()
    vector = embeddings.embed_query(user_query)
    
    # Step 2: Query the Pinecone index with the selected namespace
    search_results = index.query(
        namespace=namespace,  # Use the selected namespace
        vector=vector,
        top_k=5,  # Retrieve the top 5 most relevant matches
        include_values=False,
        include_metadata=True
    )
    
    # Step 3: Handle search results and construct response
    if search_results and "matches" in search_results and search_results["matches"]:
        # Extract relevant text from each match
        answers = [
            match["metadata"].get("text", "No relevant content found.")
            for match in search_results["matches"]
        ]
        print("Extracted Answers:", answers)
        return answers  # Return the list of content for quiz generation
    
    print("No matches found.")
    return []


def generate_quiz_from_answers(answers: List[str]) -> List[dict]:
    # Use answers to generate quiz questions
    content = "\n".join(answers)
    prompt = f"""
    Based on the following content, generate 5 multiple choice questions.
    
    Content:\n
    {content}
    
    Each question should have 4 answer options with only one correct answer.
    Format the output as a list of dictionaries with keys: 'question', 'choices', 'correct_answer'.
    """
    response = model_internquest.generate_content(prompt)
    return json.loads(response.text)

def Quiz_page():
    st.header("🧠 Knowledge Quiz")
    
    # Step 1: Add a namespace selection input
    namespace_options = ["pgbi-ikm", "pgbi-digital-leadership"]
    selected_namespace = st.selectbox("Select Namespace:", namespace_options)
    
    # User inputs a query for quiz generation
    user_query = st.text_input("Enter a topic or question to generate your quiz:")
    
    if user_query:
        with st.spinner("Fetching data..."):
            # Step 2: Fetch top 5 answers from Pinecone with the selected namespace
            top_answers = get_top_5_answers(user_query, selected_namespace)
            
            if not top_answers:
                st.error("No relevant data found. Please try a different query.")
                return
            
            # Step 3: Generate quiz questions using Google Gemini
            quiz_data_list = generate_quiz_from_answers(top_answers)
            
            if not quiz_data_list:
                st.error("Failed to generate quiz questions. Please try again.")
                return

            st.success("Quiz Generated! Let's begin.")
            
            # Step 4: Display the Quiz
            st.subheader("📝 Answer the following questions:")

            if 'user_answers' not in st.session_state:
                st.session_state.user_answers = [None] * len(quiz_data_list)
                st.session_state.correct_answers = []
                st.session_state.randomized_options = []

                for q in quiz_data_list:
                    options = q['choices']
                    st.session_state.randomized_options.append(options)
                    st.session_state.correct_answers.append(q['correct_answer'])

            with st.form(key='quiz_form'):
                for i, q in enumerate(quiz_data_list):
                    options = st.session_state.randomized_options[i]
                    default_index = 0
                    user_response = st.radio(q['question'], options, index=default_index, key=f"q_{i}")
                    st.session_state.user_answers[i] = user_response

                results_submitted = st.form_submit_button(label='Submit Answers')
                
                if results_submitted:
                    score = sum([
                        ua == st.session_state.correct_answers[i]
                        for i, ua in enumerate(st.session_state.user_answers)
                    ])
                    st.success(f"Your Score: {score}/{len(quiz_data_list)}")

                    if score == len(quiz_data_list):
                        st.balloons()
                    else:
                        incorrect_count = len(quiz_data_list) - score
                        st.warning(f"You got {incorrect_count} question(s) wrong. Let's review them:")

                    for i, (ua, ca, q) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers, quiz_data_list)):
                        with st.expander(f"Review Question {i + 1}", expanded=False):
                            if ua != ca:
                                st.error(f"Your answer: {ua}")
                                st.success(f"Correct answer: {ca}")


if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Home'

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color: #FFFFFF;
#     }
# </style>
# """, unsafe_allow_html=True)

with st.sidebar:

    selected_tab =sac.menu([
    sac.MenuItem('Home', icon='house-fill'),
    sac.MenuItem('CogniAssist',icon='book-half'),
    sac.MenuItem('Quiz',icon='patch-question-fill')
    
], color='cyan', size='lg', open_all=True)
if selected_tab != st.session_state.current_tab:
        st.session_state.current_tab = selected_tab

 
if st.session_state.current_tab == 'Home':
    home_page()
elif st.session_state.current_tab == 'CogniAssist':
    show_learning_assistant()  
elif st.session_state.current_tab == 'Quiz':
    Quiz_page()