import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
import re

st.set_page_config(
    page_title="Engineering Admissions Copilot - JoSSA 2025",
    page_icon="🎓",
    layout="wide"
)

# Setup Gemini

# Use secret from .streamlit/secrets.toml
import google.generativeai as genai
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

cutoffs = pd.read_csv("cutoffs2024.csv")

institutes = sorted(cutoffs['Institute'].dropna().unique())
branches = sorted(cutoffs['Branch'].dropna().unique())
genders = sorted(cutoffs['Gender'].dropna().unique())

# Map common abbreviations to full branch keyword
branch_map = {
    "cs": "computer",
    "cse": "computer",
    "computer science": "computer",
    "computers": "computer",
    "ece": "electronics and communication",
    "ee": "electrical",
    # Add more if needed
}

institute_map = {
    "iit": "indian institute of technology",
    "iits": "indian institute of technology",
    "nit": "national institute of technology",
    "nits": "national institute of technology",
    "iiit": "indian institute of information technology",
    "iiits": "indian institute of information technology",
    # Add more if needed
}

# Streamlit UI
st.subheader("🎓 Engineering Admissions Copilot - JoSSA 2025")
#st.subheader("JoSSA 2025 predictions based on 2024 cutoff data")

st.markdown(
    """
    <p style="font-size:16px; color:gray; text-align:center;">
    This tool is an open-source initiative to help organize JoSAA 2024 cutoff data for easier exploration.  
    All data is used as-is and may contain errors. Use this tool at your own risk. The author is not liable for any inaccuracies or decisions based on this data.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.form("form"):
    st.subheader("Student Profile")
    crl = st.number_input("Enter your JEE Main Rank", min_value=1)
    category = st.selectbox("Category", ["OPEN", "OPEN (Pwd)", "OBC-NCL", "OBC-NCL (PwD)", "SC", "SC (PwD)", "ST", "ST (PwD)", "EWS", "EWS (PwD)"])
    gender = st.selectbox("Gender", genders, index=1)
    #state = st.text_input("Domicile State")
    round_selected = st.selectbox("Select JoSAA Round", ["ANY", 1, 2, 3, 4, 5], index=1)
    selected_institute = st.selectbox("Filter by Institute", ["All", "IITs", "NITs"] + institutes)
    branch_query = st.text_input("Filter by Branch (comma-separated) (for example: cs, ece, electrical, civil)", "")
    submit = st.form_submit_button("Find Colleges")

if submit:
    # Filter based on CRL, category
    matches = cutoffs[
        (cutoffs['Closing Rank'] >= crl) &
        (cutoffs['Category'].str.lower() == category.lower()) &
        (cutoffs['Gender'].str.lower() == gender.lower())
    ]
    if round_selected != "ANY":
        matches = matches[matches['Round'] == round_selected]

    # Apply optional institute filter
    if selected_institute != "All":
        selected_institute_normalized = selected_institute.strip().lower()
        # Replace if found in map
        if selected_institute_normalized in institute_map:
            selected_institute = institute_map[selected_institute_normalized]
        else:
            selected_institute = selected_institute_normalized
        #st.text("Expanded Institute Name:")
        #st.text(selected_institute)
        matches = matches[matches['Institute'].str.lower().str.contains(selected_institute.lower(), regex=False)]


    # Apply optional branch filter
    if branch_query.strip() != "":
        # Normalize and split input
        branch_keywords = [kw.strip().lower() for kw in branch_query.split(",") if kw.strip()]

        branch_keywords = [branch_map.get(kw, kw) for kw in branch_keywords]
        
        # Apply filter: match if any keyword appears in the branch name
        pattern = '|'.join(branch_keywords)
        #st.text("Branch Pattern:")
        #st.text(pattern)
        matches = matches[matches['Branch'].str.lower().str.contains(pattern)]

    # Drop duplicates based on key columns
    matches_unique = matches.drop_duplicates(subset=['Institute', 'Branch', 'Category'])

    if matches_unique.empty:
        st.warning("⚠️ Sorry, no colleges found for your profile.")
    else:
        st.success(f"🎯 Found {len(matches_unique)} possible options based on 2024 cutoffs!")

        # Select only relevant columns
        display_data = matches_unique[['Closing Rank', 'Institute', 'Branch', 'Round']].sort_values(by='Closing Rank')

        # Reset index and convert to records for clean display
        display_data = display_data.reset_index(drop=True)

        # Convert to a format that Streamlit doesn't try to add index to
        st.dataframe(display_data.style.hide(axis='index'), use_container_width=True)



st.markdown("---")  # Horizontal line separator


# Gemini assistant

# Suggested example questions
sample_questions = [
    "What NITs can I get with 15000 rank for ECE?",
    "Show me IITs accepting 8000 rank for Computer Science.",
    "What are my options in Round 3 with rank 23000 for SC category?",
    "Can I get Mechanical in NIT with 12000 rank?",
    "What branches are available in IITs above 6000 CRL?"
]

# UI section for hybrid input
st.subheader("🤖 Ask Admissions Copilot")

st.markdown(
    """
    <p style="font-size:16px; color:gray; text-align:center;">
    This section is an attempt to provide the same information using GenAI by understanding an English query.
    The system is able to understand simple queries with one branch, one college type and a rank but may fail at complex ones. Give it a try!
    </p>
    """,
    unsafe_allow_html=True,
)

selected_example = st.selectbox("💡 Choose a sample question:", [""] + sample_questions)
custom_question = st.text_area("Or type your own question below:")

# Final question used for processing
final_question = custom_question.strip() if custom_question.strip() else selected_example


def trigger_gemini():
    st.session_state["last_question"] = final_question
    st.session_state["run_query"] = True

st.button("Ask", on_click=trigger_gemini)

if st.session_state.get("run_query", False) and st.session_state.get("last_question", "").strip():
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        You're an Indian college admission counselor.
        You are helping students get admission into engineering colleges via JoSAA.

        Understand:
        - "IIT" = Indian Institute of Technology (e.g. IIT Bombay)
        - "NIT" = National Institute of Technology (e.g. NIT Trichy)
        - "IIIT" = Indian Institute of Information Technology (not same as IIT)

        Based on this student question, extract the following fields:
        - Closing Rank (numeric)
        - Round (1-5)
        - Category (OPEN, OBC, SC, etc.)
        - Branch (text)
        - Institute (text)

        Respond ONLY with raw JSON (no markdown/code blocks).

        Question: {st.session_state['last_question']}
        """

        response = model.generate_content(prompt)
        output = response.text
        clean_output = re.sub(r"```json|```", "", output).strip()

        extracted = json.loads(clean_output)
        #st.success("✅ Parsed Filters:")
        #st.json(extracted)

        # Filter data
        extracted = {k.lower(): v for k, v in extracted.items()}
        matches = cutoffs.copy()

        if "closing rank" in extracted and isinstance(extracted["closing rank"], int):
            matches = matches[matches["Closing Rank"] >= extracted["closing rank"]]

        if "round" in extracted and isinstance(extracted["round"], int):
            matches = matches[matches["Round"] == extracted["round"]]

        if "category" in extracted and isinstance(extracted["category"], str):
            cat_filter = extracted["category"]
            cat_filter = cat_filter.replace("General", "OPEN")
            cat_filter = cat_filter.replace("GEN", "OPEN")
            #st.text("Category Name:")
            #st.text(cat_filter)
            matches = matches[matches["Category"].str.upper() == extracted["category"].upper()]
        else:
            matches = matches[matches["Category"].str.upper() == "OPEN"]


        if "branch" in extracted and isinstance(extracted["branch"], str):
            branch_filter = extracted["branch"]
            branch_filter_normalized = branch_filter.strip().lower()
            # Replace if found in map
            if branch_filter_normalized in branch_map:
                branch_filter = branch_map[branch_filter_normalized]
            else:
                branch_filter = branch_filter_normalized
            #st.text("Expanded Branch Name:")
            #st.text(branch_filter)
            matches = matches[matches["Branch"].str.contains(branch_filter, case=False, na=False)]

        if "institute" in extracted and isinstance(extracted["institute"], str):
            inst_filter = extracted["institute"]
            inst_filter_normalized = inst_filter.strip().lower()
            # Replace if found in map
            if inst_filter_normalized in institute_map:
                inst_filter = institute_map[inst_filter_normalized]
            else:
                inst_filter = inst_filter_normalized
            #st.text("Expanded Institute Name:")
            #st.text(inst_filter)
            matches = matches[matches["Institute"].str.contains(inst_filter, case=False, na=False)]

        matches_unique = matches.drop_duplicates(subset=['Institute', 'Branch', 'Category'])

        if not matches_unique.empty:
            st.success(f"🎓 Found {len(matches_unique)} matching options based on 2024 cutoffs!")
            display_data = matches_unique[['Closing Rank', 'Institute', 'Branch', 'Category', 'Round']].sort_values(by='Closing Rank')
            display_data = display_data.reset_index(drop=True)
            st.dataframe(display_data.style.hide(axis='index'), use_container_width=True)
        else:
            st.warning("No matches found. Try refining your question.")

    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Parse Error: {e}")
        st.text("Raw model output:")
        st.text(output)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

    # Reset trigger
    st.session_state["run_query"] = False


st.markdown("---")  # Horizontal line separator

st.markdown(
    """
    <p style="font-size:12px; color:gray; text-align:center;">
    &copy; 2025 Ritesh Jain. This tool is an open-source initiative to help organize JoSAA 2024 cutoff data for easier exploration.  
    All data is used as-is and may contain errors. Use this tool at your own risk. The author is not liable for any inaccuracies or decisions based on this data.
    </p>
    """,
    unsafe_allow_html=True,
)

