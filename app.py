import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
import os



# ---- Setup ----
st.set_page_config(page_title="AI Case Viability Advisor", page_icon="⚖️")
st.title("⚖️ AI Case Viability Advisor")
st.write("Upload your case description and documents. Get an expert report on viability, strengths, risks, and recommendations.")


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()



# ---- LangChain Model Setup ----
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    streaming=True,
    temperature=0.1,
    model="gpt-4o"  # or "gpt-3.5-turbo"
)


# ---- Intake Form ----
with st.form("case_form"):
    case_summary = st.text_area("Describe your case (what happened, who is involved, what outcome do you seek?)", height=200)
    uploaded_files = st.file_uploader(
        "Upload relevant documents (contracts, correspondence, evidence, etc.)",
        type=["pdf", "docx", "txt", "png", "jpg"],
        accept_multiple_files=True,
    )
    submit = st.form_submit_button("Analyze Case")

# ---- Process on Submit ----
if submit and case_summary:
    st.info("Analyzing your case, please wait...")

    # For POC, we'll just use the text summary. (Files can be parsed and injected into the prompt in next iterations.)
    # Advanced: Parse PDFs/DOCs to extract text and add to prompt as "evidence"

    prompt = f"""
You are an expert legal case advisor specializing in UAE law and judicial practice.
Given the following client intake, provide a comprehensive report with:

1. Executive summary (Go/No-Go/Consider, probability of success, key risks).
2. Case overview (type, parties, timeline).
3. Legal issues and relevant law (UAE statutes and verdicts if possible).
4. Evidence assessment.
5. Predictive analysis (likelihoods, uncertainties).
6. Reasoned legal analysis.
7. Concrete recommendations (actions to strengthen the case, missing info).
8. Disclaimer (limit to what was provided).
Output should be in structured Markdown.

Client Intake:
---
{case_summary}
"""

    # Stream output from OpenAI using LangChain
    response_container = st.empty()
    full_response = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if isinstance(chunk, AIMessage):
            full_response += chunk.content
            response_container.markdown(full_response)

    # (Optional) Display uploaded files
    if uploaded_files:
        st.subheader("Uploaded Documents")
        for f in uploaded_files:
            st.write(f"- {f.name}")

else:
    st.warning("Please enter a case summary to get started.")

# ---- Footer ----
st.markdown("---")
st.caption("Prototype – No legal advice. Powered by OpenAI + LangChain. For demonstration only.")