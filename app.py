import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Set page configuration
st.set_page_config(
    page_title="Email Generator",
    page_icon="📧",
    layout="wide"
)

# Session state initialization
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

def load_all_data():
    """Load all datasets and return them as a dictionary"""
    try:
        linkedin_data = pd.read_csv("data/linkedin_data.csv")
        website_data = pd.read_csv("data/website_data.csv")
        news_data = pd.read_csv("data/news_data.csv")
        
        return {
            "linkedin": linkedin_data,
            "website": website_data,
            "news": news_data
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_person_info(person_name, all_data):
    """Get comprehensive information about a person from all datasets"""
    try:
        # Get LinkedIn data for the person
        linkedin_info = all_data["linkedin"][all_data["linkedin"]["name"] == person_name].iloc[0].to_dict()
        
        # Get company name from LinkedIn data
        company_name = linkedin_info["company"]
        
        # Get website data for the company
        website_info = all_data["website"][all_data["website"]["company_name"] == company_name].iloc[0].to_dict()
        
        # Get news data related to the company
        news_info = all_data["news"][all_data["news"]["related_company"] == company_name].iloc[0].to_dict()
        
        # Combine all information
        comprehensive_info = {
            "person": linkedin_info,
            "company": website_info,
            "news": news_info
        }
        
        return comprehensive_info
    except Exception as e:
        st.error(f"Error retrieving comprehensive information: {str(e)}")
        return None

def generate_email(data, prompt):
    """Generate email using Gemini LLM based on selected data and prompt"""
    try:
        # Configure the model with safety settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            # generation_config=generation_config,
            # safety_settings=safety_settings
        )
        
        # Format the data as a string for the prompt
        data_str = ""
        for category, info in data.items():
            data_str += f"\n--- {category.upper()} INFORMATION ---\n"
            data_str += "\n".join([f"{key}: {value}" for key, value in info.items()])
            data_str += "\n"
        
        full_prompt = f"""
        Based on the following data:
        
        {data_str}
        
        And following these instructions:
        {prompt}
        
        Generate a professional email with:
        1. A compelling subject line
        2. A personalized greeting
        3. Email body that incorporates the data effectively
        4. A professional closing
        
        Format the response as:
        
        Subject: [Subject Line]
        
        [Email Body]
        
        [Signature]
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating email: {str(e)}"

def main():
    st.title("📧 Email Generator")
    st.subheader("Generate personalized emails using AI and comprehensive data")
    
    # API Key input section
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input("Enter your Google API Key:", type="password")
        if st.button("Configure API"):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    st.session_state.api_key_configured = True
                    st.success("API key configured successfully!")
                except Exception as e:
                    st.error(f"Error configuring API key: {str(e)}")
            else:
                st.error("Please enter a valid API key")
    
    # Main app content - only show if API key is configured
    if not st.session_state.api_key_configured:
        st.info("Please configure your Google API key in the sidebar to use the application.")
        st.markdown("""
        ### How to get a Google API Key:
        1. Go to [Google AI Studio](https://makersuite.google.com/)
        2. Sign in with your Google account
        3. Create a new API key
        4. Copy and paste the key in the sidebar
        """)
        return
    
    # Load all datasets
    all_data = load_all_data()
    
    if all_data is not None:
        # Get list of people from LinkedIn data
        people_list = all_data["linkedin"]["name"].tolist()
        
        # Person selection
        selected_person = st.selectbox("Select a person:", people_list)
        
        if selected_person:
            # Get comprehensive information about the selected person
            person_info = get_person_info(selected_person, all_data)
            
            if person_info:
                # Display person summary
                st.subheader("Person Summary")
                
                # Create a concise summary of the person
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {person_info['person']['name']}")
                    st.markdown(f"**Title:** {person_info['person']['title']}")
                    st.markdown(f"**Company:** {person_info['person']['company']}")
                    st.markdown(f"**Location:** {person_info['person']['location']}")
                
                with col2:
                    st.markdown("### About")
                    st.markdown(person_info['person']['about'])
                
                # Tabs for detailed information
                tab1, tab2, tab3 = st.tabs(["LinkedIn Profile", "Company Information", "Recent News"])
                
                with tab1:
                    st.json(person_info["person"])
                
                with tab2:
                    st.json(person_info["company"])
                
                with tab3:
                    st.json(person_info["news"])
                
                # Prompt for email generation
                st.subheader("Email Generation")
                email_prompt = st.text_area(
                    "Enter prompt for email generation:",
                    height=150,
                    placeholder="Example: Write a networking email to connect with this person, mentioning their recent company news and asking for a meeting to discuss potential collaboration."
                )
                
                # Generate email
                if st.button("Generate Email"):
                    if email_prompt:
                        with st.spinner("Generating email..."):
                            email_content = generate_email(person_info, email_prompt)
                            st.subheader("Generated Email")
                            st.text_area("", email_content, height=400)
                            
                            # Download button
                            st.download_button(
                                label="Download Email",
                                data=email_content,
                                file_name="generated_email.txt",
                                mime="text/plain"
                            )
                    else:
                        st.warning("Please enter a prompt for email generation.")
            else:
                st.error("Could not retrieve comprehensive information for the selected person.")
    else:
        st.error("Could not load data. Please check if the CSV files exist in the data directory.")

if __name__ == "__main__":
    main()
