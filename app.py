# -------------------------------
# STEP 1: Libraries import kar rahe hain
# -------------------------------

import streamlit as st                    # Web app banane ke liye
import PyPDF2                             # PDF resume read karne ke liye
from sklearn.feature_extraction.text import CountVectorizer   # Text ko numbers mein convert karne ke liye
from sklearn.metrics.pairwise import cosine_similarity        # Resume & job match calculate karne ke liye


# -------------------------------
# STEP 2: Page settings + Title
# -------------------------------

st.set_page_config(page_title="AI Resume Analyzer")

st.title("📄 AI Resume Analyzer")
st.write("Resume upload karo aur job description paste karo")


# -------------------------------
# STEP 3: User se input lena
# -------------------------------

# Resume upload (sirf PDF allow hai)
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job description input
job_desc = st.text_area("Paste Job Description here")


# -------------------------------
# STEP 4: Resume PDF se text nikalne ka function
# -------------------------------

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)   # PDF reader object
    text = ""                             # Empty string

    # Resume ke har page ka text add kar rahe hain
    for page in reader.pages:
        text += page.extract_text()

    return text                           # Final resume text return


# -------------------------------
# STEP 5: Analyze button ka logic
# -------------------------------

if st.button("Analyze Resume"):

    # Check: resume aur job description dono diye gaye hain?
    if uploaded_file and job_desc:

        # Resume text read karo
        resume_text = extract_text(uploaded_file)

        # Resume + Job description ko ek list mein daal do
        documents = [resume_text, job_desc]

        # Text ko vector (numbers) mein convert kar rahe hain
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(documents)

        # Similarity calculate karo (percentage mein)
        match_percentage = cosine_similarity(vectors)[0][1] * 100

        # Result show karo
        st.success(f"Resume Match Score: {round(match_percentage, 2)}%")

        # Result ke hisaab se message
        if match_percentage > 75:
            st.balloons()
            st.write("✅ Excellent match! You can apply confidently.")
        elif match_percentage > 50:
            st.warning("⚠ Good match, but improve some skills.")
        else:
            st.error("❌ Low match. Resume improve karo.")

    else:
        st.error("Please upload resume AND paste job description.")
