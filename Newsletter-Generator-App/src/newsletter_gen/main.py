#!/usr/bin/env python
import sys
from newsletter_gen.crew import NewsletterGenCrew

def load_html_template():
    with open(r'Newsletter-Generator-App/src/newsletter_gen/config/newsletter_template.html', 'r') as file:
        html_template = file.read()
    return html_template

def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': input("Enter a topic for your news letter: "),
        'personal_message': input("Enter a personal message for your news letter: "),
        'html_template': load_html_template()
    }
    NewsletterGenCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        NewsletterGenCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
