import streamlit as st
import sys
import os
from newsletter_gen.crew import NewsletterGenCrew

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

class NewsletterGenUI:
    def load_html_template(self):
        with open(r'Newsletter-Generator-App/src/newsletter_gen/config/newsletter_template.html', 'r') as file:
            html_template = file.read()
        return html_template

    def generate_newsletter(self, topic, personal_message):
        inputs = {
            'topic': topic,
            'personal_message': personal_message,
            'html_template': self.load_html_template()
        }
        return NewsletterGenCrew().crew().kickoff(inputs=inputs)

    def newsletter_generation(self):
        if st.session_state.generating:
            st.session_state.newsletter = self.generate_newsletter(
                st.session_state.topic, st.session_state.personal_message,
            )
        if st.session_state.newsletter and st.session_state.newsletter != "":
            with st.container():
                st.write("Newsletter generated successfully!")
                st.download_button(
                    label="Download HTML file",
                    data=st.session_state.newsletter,
                    file_name="newsletter.html",
                    mime="text/html",
                )
            st.session_state.generating = False
                
    def sidebar(self):
        with st.sidebar:
            st.title("Newsletter Generator")

            st.write(
                """
                To generate a newsletter, enter a topic and a personal message. \n
                Your team of AI agents will generate a newsletter for you!
                """
            )
    
            topic = st.text_input("Topic", key = "topic", placeholder="USA Stock Market")
            
            st.text_area(
                "Your personal message (This would be included at the top of the newsletter)", key = "personal_message", 
                placeholder="Hello, welcome to this week's newsletter!")
            
            if st.button("Generate Newsletter"):
                st.session_state.generating = True
                
    def render(self):
        st.set_page_config(page_title="Newsletter Generation", page_icon="📰")
        if "topic" not in st.session_state:
            st.session_state.topic = ""

        if "personal_message" not in st.session_state:
            st.session_state.personal_message = ""
        
        if "newsletter" not in st.session_state:
            st.session_state.newsletter = ""
        
        if "generating" not in st.session_state:
            st.session_state.generating = False
        self.sidebar()
        self.newsletter_generation()

if __name__ == "__main__":
    NewsletterGenUI().render()
