import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
gb_model = joblib.load('GB.pkl')  # Ensure the model file is in the same directory

# Load the dataset and drop unnecessary columns
data = pd.read_csv("data.csv").drop(columns=['Unnamed: 0'])  # Drop extra columns

# Function to recommend learning path
def recommend_learning_path(student_id, df):
    # Check if the student ID exists in the dataset
    if student_id not in df['id_student'].values:
        return "Student not found.", None, None

    # Get the data for the specific student
    student_data = df[df['id_student'] == student_id]

    # Drop unnecessary columns for prediction
    student_data = student_data.drop(columns=['id_student', 'study_method_preference'])

    # Ensure the columns match the model's training data
    student_data = student_data[gb_model.feature_names_in_]

    # Predict the study method preference
    predicted_label = gb_model.predict(student_data)

    # Extract engagement level
    engagement = student_data["engagement_classification"].iloc[0]

    # Recommendations based on study method and engagement
    recommendations = {
        0: {  # Collaborative
            0: ["Interactive AI Basics: Weekly Quizzes and Forums", "Applied AI: Practical Exercises with Peer Feedback",
                "Introduction to Machine Learning: Online Workshops", "AI Ethics: Case Studies and Discussion Groups"],  # Moderate Engagement
            1: ["Collaborative AI Projects: Team-Based Learning", "Advanced AI Techniques: Group Workshops and Peer Reviews",
                "Machine Learning Bootcamp: Intensive Group Projects", "AI in Practice: Team Challenges and Hackathons"],  # High Engagement
            2: ['Introduction to AI: Self-Paced Fundamentals', 'AI Basics: Introductory Video Series',
                'Foundations of Machine Learning: Self-Study Edition', 'AI for Everyone: Introductory Readings and Quizzes']  # Low Engagement
        },
        1: {  # Offline Content
            0: ["AI Principles: Self-Study with Case Studies", "Machine Learning: Offline Course with Practice Problems",
                "Applied AI: Textbook and Supplementary Materials", "Data Science: Case Studies and Analytical Exercises"],  # Moderate Engagement
            1: ["Advanced AI: Comprehensive Textbook with Projects", "Deep Learning: In-Depth Study with Capstone Projects",
                "AI and Machine Learning: Project-Based Learning", "Data Science Mastery: Offline Content with Comprehensive Projects"],  # High Engagement
            2: ['AI Basics: Essential Readings and Key Concepts', 'Machine Learning Fundamentals: Self-Study Workbook',
                "AI Concepts: Downloadable Lecture Series", "Introduction to Data Science: Offline Learning Modules"]  # Low Engagement
        },
        2: {  # Interactive
            0: ["Machine Learning: Interactive Coding Exercises", "AI Applications: Interactive Case Studies",
                "Data Science: Interactive Projects and Peer Reviews", "AI Ethics: Discussion Forums and Interactive Scenarios"],  # Moderate Engagement
            1: ["Advanced AI: Interactive Group Projects and Hackathons", "Deep Learning: Interactive Labs and Collaborative Projects",
                "Machine Learning Mastery: Interactive Workshops and Challenges", "AI Research: Collaborative Research Projects and Peer Feedback"],  # High Engagement
            2: ["AI Basics: Interactive Quizzes and Flashcards", "Introduction to Machine Learning: Interactive Visualizations",
                "AI Fundamentals: Interactive Notebooks", "AI Concepts: Gamified Learning Modules"]  # Low Engagement
        },
        3: {  # Informational
            0: ["Machine Learning: Structured Video Course", "AI Concepts: Comprehensive Video Series",
                "Data Science: Interactive Reading and Video Modules", "AI in Practice: Lecture Notes and Case Studies"],  # Moderate Engagement
            1: ["Advanced AI: Detailed Lecture Series and Readings", "Deep Learning: Advanced Lecture Series with Supplemental Readings",
                "AI and Machine Learning: Research Papers and Advanced Lectures", "Data Science Masterclass: Comprehensive Reading and Video Content"],  # High Engagement
            2: ["AI Overview: Short Video Lectures", "Introduction to Machine Learning: Podcast Series",
                "AI Fundamentals: Infographics and Summaries", "Data Science: Essential Readings and Articles"]  # Low Engagement
        },
        4: {  # Resource-Based
            0: ["Machine Learning: Comprehensive eBooks and Guides", "AI Applications: Case Study Compilations",
                "Data Science: In-Depth Articles and White Papers", "AI Concepts: Research Articles and Detailed Guides"],  # Moderate Engagement
            1: ["Advanced AI: Research Papers and Technical Reports", "Deep Learning: Comprehensive Textbooks and Resource Repositories",
                "Machine Learning Mastery: Advanced Documentation and APIs", "AI Ethics: Government and Institutional Reports"],  # High Engagement
            2: ["AI Basics: Curated Reading Lists", "Introduction to Machine Learning: Beginner-Friendly Blogs",
                "Data Science Overview: Quick Reference Guides", "AI Fundamentals: Online Documentation"]  # Low Engagement
        }
    }

    # Determine study method and engagement level
    study_method = predicted_label[0]
    engagement_level = engagement

    # Get the recommended courses based on study method and engagement level
    recommended_courses = recommendations.get(study_method, {}).get(engagement_level, [])

    return recommended_courses, study_method, engagement_level

# Function to map numeric preferences to their original labels
def return_map_to_original_preference(x):
    if x == 0:
        return 'Collaborative'
    elif x == 1:
        return 'Offline Content'
    elif x == 2:
        return 'Interactive'
    elif x == 3:
        return 'Informational'
    elif x == 4:
        return 'Resource-Based'

# Function to map numeric engagement levels to their original labels
def return_map_to_original_engagement(x):
    if x == 0:
        return 'Moderate Engagement'
    elif x == 1:
        return 'High Engagement'
    elif x == 2:
        return 'Low Engagement'

# Streamlit app
def main():
    st.title('Learning Path Recommendation Engine')


    # Input for student ID
    student_id = st.number_input("Enter Student ID", min_value=0, step=1)

    if st.button("Get Recommendations"):
        recommendations, study_method, engagement = recommend_learning_path(student_id, data)
        if recommendations == "Student not found.":
            st.error("Student not found.")
        else:
            study_method = return_map_to_original_preference(study_method)
            engagement = return_map_to_original_engagement(engagement)

            st.write(f"**Study method preference for student {student_id}:** {study_method}")
            st.write(f"**Level of engagement for student {student_id}:** {engagement}")
            st.write("**Recommended courses:**")
            for course in recommendations:
                st.write(f"- {course}")

if __name__ == "__main__":
    main()