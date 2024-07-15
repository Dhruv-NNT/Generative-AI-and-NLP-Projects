# 📬 Crew Automation Newsletter Generation with GUI

Welcome to the Crew Automation Newsletter Generation project! This project leverages CrewAI to build a team of autonomous agents that generate newsletters. The graphical user interface (GUI) allows users to input a topic and personal message, and then displays the thought process of each agent as they work on the newsletter. Once the automation is complete, users can download the HTML newsletter.

## 📝 Project Overview

This project utilizes CrewAI, a framework that allows you to build teams of autonomous agents, to generate newsletters. The automation employs a group of research agents to locate pertinent news articles, a chief editor to revamp the titles and summaries, and an HTML writer to design the newsletter. The GUI enables users to input a topic and a personal message, and then it displays the thoughts of each agent as they work on the newsletter. Once the automation is complete, users can download the HTML newsletter.

## 🌟 Features

- **Automated Newsletter Generation**: Generates newsletters by finding relevant news articles based on user-provided topic and personal message.
- **Agent Thought Process Display**: Displays the thought process of each agent involved in the automation process (researching, editing, writing).
- **Downloadable HTML**: Allows users to download the generated HTML newsletter.

## 🛠️ Technologies Used

- **CrewAI**
- **Python**
- **Streamlit** (for building the GUI)

## 📂 Project Structure

```
📦Crew Automation Newsletter Generation
 ┣ 📂src
 ┃ ┣ 📂newsletter_gen
 ┃ ┃ ┣ 📜agents.yaml        # Configuration for agents
 ┃ ┃ ┣ 📜tasks.yaml         # Configuration for tasks
 ┃ ┃ ┣ 📜crew.py            # Logic, tools, and specific args for agents
 ┃ ┃ ┗ 📜main.py            # Custom inputs for agents and tasks
 ┣ 📜.env                   # Environment variables (e.g., OPENAI_API_KEY)
 ┣ 📜README.md              # Project overview and instructions
 ┣ 📜pyproject.toml         # Poetry configuration file
 ┗ 📜run.sh                 # Shell script to run the application
```

## ⚙️ Installation

Ensure you have Python >=3.10 <=3.13 installed on your system. This project uses [Poetry](https://python-poetry.org/) for dependency management and package handling.

First, if you haven't already, install Poetry:

```bash
pip install poetry
```

Next, navigate to your project directory and install the dependencies:

1. Lock the dependencies and then install them:

```bash
poetry lock
```

```bash
poetry install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/newsletter_gen/config/agents.yaml` to define your agents.
- Modify `src/newsletter_gen/config/tasks.yaml` to define your tasks.
- Modify `src/newsletter_gen/crew.py` to add your own logic, tools, and specific args.
- Modify `src/newsletter_gen/main.py` to add custom inputs for your agents and tasks.

## 🚀 Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
poetry run newsletter_gen
```

This command initializes the newsletter-gen Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will create a `report.md` file with the output of a research on LLMs in the root folder.

## 📊 Understanding Your Crew

The newsletter-gen Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## 📧 Support

For support, questions, or feedback regarding the NewsletterGen Crew or CrewAI:
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of CrewAI.

---

Feel free to further customize this README file based on any additional specific details or preferences you have!
