from utils import *
from tools import *
from openai import OpenAI
import os
import json
import re
import tiktoken
from token_processor import token_processor

def extract_json_between_markers(llm_output):
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue
    return None

def query_model(model_str, system_prompt, prompt, temp=None, openai_api_key=None):
    if temp is None:
        temp = 0.7 if "gpt-4" in model_str else 0.0
    
    if model_str.startswith("lmstudio-"):
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )
        model_name = model_str.replace('lmstudio-', '')
    elif model_str.startswith("groq-"):
        groq_api = os.getenv('GROQ_API_KEY')
        if not groq_api:
            raise Exception("No Groq API key provided")
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api
        )
        model_name = model_str.replace('groq-', '')
    elif model_str.startswith("ollama-"):
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        model_name = model_str.replace('ollama-', '')
    else:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        client = OpenAI()
        model_name = model_str

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    messages = token_processor.batch_process_messages(messages, model_name)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content
    except Exception as e:
        if model_str.startswith("lmstudio-"):
            print(f"\nError with LM Studio: {str(e)}")
            print("Please ensure:")
            print(f"1. LM Studio server is running at http://localhost:1234")
            print(f"2. Model '{model_name}' is downloaded and loaded")
        elif model_str.startswith("ollama-"):
            print(f"\nError with Ollama: {str(e)}")
            print("Please ensure:")
            print("1. Ollama is running at http://localhost:11434")
            print(f"2. Model '{model_name}' is pulled and available")
        raise

def get_score(outlined_plan, latex, reward_model_llm, reviewer_type=None, attempts=3, openai_api_key=None):
    e = str()
    for _attempt in range(attempts):
        try:
            template_instructions = """
            Respond in the following format:

            THOUGHT:
            <THOUGHT>

            REVIEW JSON:
            ```json
            <JSON>
            ```

            In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
            Detail your high-level arguments, necessary choices and desired outcomes of the review.
            Do not make generic comments here, but be specific to your current paper.
            Treat this as the note-taking phase of your review.

            In <JSON>, provide the review in JSON format with the following fields in the order:
            - "Summary": A summary of the paper content and its contributions.
            - "Strengths": A list of strengths of the paper.
            - "Weaknesses": A list of weaknesses of the paper.
            - "Originality": A rating from 1 to 4 (low, medium, high, very high).
            - "Quality": A rating from 1 to 4 (low, medium, high, very high).
            - "Clarity": A rating from 1 to 4 (low, medium, high, very high).
            - "Significance": A rating from 1 to 4 (low, medium, high, very high).
            - "Questions": A set of clarifying questions to be answered by the paper authors.
            - "Limitations": A set of limitations and potential negative societal impacts of the work.
            - "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
            - "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Overall": A rating from 1 to 10 (very strong reject to award quality).
            - "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
            - "Decision": A decision that has to be one of the following: Accept, Reject.
            """

            if reviewer_type is None:
                reviewer_type = ""
            
            sys = (
                "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
                f"Be critical and cautious in your decision. {reviewer_type}\n"
            ) + template_instructions

            scoring = query_model(
                model_str=f"{reward_model_llm}",
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined in the following text is the research plan that the machine learning engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research latex that the model produced: \n{latex}\n\n"
                ),
                temp=0.0
            )

            review_json = extract_json_between_markers(scoring)

            overall = int(review_json["Overall"]) / 10
            soundness = int(review_json["Soundness"]) / 4
            confidence = int(review_json["Confidence"]) / 5
            contribution = int(review_json["Contribution"]) / 4
            presentation = int(review_json["Presentation"]) / 4
            clarity = int(review_json["Clarity"]) / 4
            originality = int(review_json["Originality"]) / 4
            quality = int(review_json["Quality"]) / 4
            significance = int(review_json["Significance"]) / 4

            clarity_weight = 0.1
            quality_weight = 0.1
            overall_weight = 1.0
            soundness_weight = 0.1
            confidence_weight = 0.1
            originality_weight = 0.1
            significance_weight = 0.1
            contribution_weight = 0.4
            presentation_weight = 0.2

            max_score = (
                clarity_weight + quality_weight + overall_weight + soundness_weight +
                confidence_weight + originality_weight + significance_weight +
                contribution_weight + presentation_weight
            )

            performance = ((
                soundness_weight * soundness +
                presentation_weight * presentation +
                confidence_weight * confidence +
                contribution_weight * contribution +
                overall_weight * overall +
                originality_weight * originality +
                significance * significance_weight +
                clarity_weight * clarity +
                quality_weight * quality
            ) / max_score) * 10

            return performance, f"The performance of your submission is: {performance}" + scoring, True

        except Exception as e:
            if _attempt < attempts - 1:
                print(f"Attempt {_attempt + 1} failed: {str(e)}. Retrying...")
                continue
            print(f"All {attempts} attempts failed. Last error: {str(e)}")
            return 0, str(e), False
    return 0, e

class ReviewersAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, openai_api_key=None):
        if notes is None:
            self.notes = []
        else:
            self.notes = notes
        self.model = model
        self.openai_api_key = openai_api_key

    def inference(self, plan, report):
        reviewer_1 = "You are a harsh but fair reviewer and expect good experiments that lead to insights for the research topic."
        review_1 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_1, openai_api_key=self.openai_api_key)

        reviewer_2 = "You are a harsh and critical but fair reviewer who is looking for an idea that would be impactful in the field."
        review_2 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_2, openai_api_key=self.openai_api_key)

        reviewer_3 = "You are a harsh but fair open-minded reviewer that is looking for novel ideas that have not been proposed before."
        review_3 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_3, openai_api_key=self.openai_api_key)

        return f"Reviewer #1:\n{review_1}, \nReviewer #2:\n{review_2}, \nReviewer #3:\n{review_3}"

class BaseAgent:
    def __init__(self, model, notes=None, max_steps=100, openai_api_key=None):
        self.model = model
        self.notes = notes if notes is not None else []
        self.max_steps = max_steps
        self.openai_api_key = openai_api_key
        self._reset_state()

    def inference(self, research_topic, phase, feedback="", step=0, temp=0.7):
        if isinstance(self.model, str) and self.model.startswith("ollama-"):
            temp = 0.7

        messages = [
            {"role": "system", "content": f"You are an AI researcher working on: {research_topic}. Current phase: {phase}. Step: {step}"},
            {"role": "user", "content": feedback}
        ]
        messages = token_processor.batch_process_messages(messages, self.model)
        
        return query_model(
            model_str=self.model,
            prompt=messages[1]["content"],
            system_prompt=messages[0]["content"],
            openai_api_key=self.openai_api_key,
            temp=temp
        )

    def _reset_state(self):
        self.phases = []
        self.lit_review = []
        self.lit_review_sum = ""
        self.dataset_code = ""
        self.plan = ""
        self.report = ""
        self.exp_results = ""
        self.results_code = ""
        self.interpretation = ""
        self.history = []

    def reset(self):
        self._reset_state()

class PhDStudentAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["research", "writing"]

    def context(self, phase):
        sr_str = ""
        if hasattr(self, 'second_round') and self.second_round:
            sr_str = (
                f"Previous Literature Review: {self.lit_review_sum}\n"
                f"Previous Plan: {self.plan}\n"
            )
        return sr_str

    def phase_prompt(self, phase):
        return (
            "You are a PhD student working on research. "
            f"Current phase: {phase}. "
            "Your goal is to conduct thorough research and produce high-quality results."
        )

    def role_description(self):
        return "a PhD student at a top university conducting research"

    def command_descriptions(self, phase):
        return (
            "You can use the following commands:\n"
            "```DIALOGUE\ndialogue here\n``` - For communication\n"
            "```SUMMARY\nsummary here\n``` - For summarizing findings\n"
        )

    def example_command(self, phase):
        return (
            "Example command usage:\n"
            "```DIALOGUE\nI think we should focus on X because...\n```\n"
            "```SUMMARY\nKey findings: 1. X improves Y by Z%...\n```\n"
        )

class ProfessorAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["report writing"]

    def generate_readme(self):
        sys_prompt = f"""You are {self.role_description()} \n Here is the written paper \n{self.report}. Task instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a readme.md for a github repository."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the readme below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp.replace("```markdown", "")

    def context(self, phase):
        #sr_str = str()
        #if self.second_round:
        #    sr_str = (
        #        f"The following are results from the previous experiments\n",
        #        f"Previous Experiment code: {self.prev_results_code}\n"
        #        f"Previous Results: {self.prev_exp_results}\n"
        #        f"Previous Interpretation of results: {self.prev_interpretation}\n"
        #        f"Previous Report: {self.prev_report}\n"
        #        f"{self.reviewer_response}\n\n\n"
        #    )
        #if phase == "report writing":
        #    return (
        #        sr_str,
        #        f"Current Literature Review: {self.lit_review_sum}\n"
        #        f"Current Plan: {self.plan}\n"
        #        f"Current Dataset code: {self.dataset_code}\n"
        #        f"Current Experiment code: {self.results_code}\n"
        #        f"Current Results: {self.exp_results}\n"
        #        f"Current Interpretation of results: {self.interpretation}\n"
        #    )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
            "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\n<Insert command here>\n``` where COMMAND is the specific command you want to run (e.g. REPORT, DIALOGUE).\n")

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "When you believe a good report has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```LATEX\nreport here\n```\n where report here is the actual report written in compilable latex to be transmitted and LATEX is just the word LATEX.\n"
            "Your report should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately. You must also submit the report promptly. Do not delay too long.\n"
            "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        phase_str = (
            "You are directing a PhD student to help them write a report in latex based on results from an experiment, and you interact with them through dialogue.\n"
            "Your goal is to write a report in latex for an experiment. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the PhD student how they can write up the results and give their feedback to improve their thoughts.\n"
        )
        return phase_str

    def role_description(self):
        return "a computer science professor at a top university."

class PostdocAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["plan formulation", "results interpretation"]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "plan formulation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}",
            )
        elif phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you believe a good plan has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```PLAN\nplan here\n```\n where plan here is the actual plan to be transmitted and PLAN is just the word PLAN. Plan here should provide a clear outline for how to achieve the task, including what machine learning models to use and implement, what types of datasets should be searched for and used to train the model, and the exact details of the experiment.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "Make sure not to produce too much dialogue and to submit an plan in reasonable time."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. PLAN, DIALOGUE).\n"
            )
        elif phase == "results interpretation":
            return (
                "When you believe a good interpretation has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```INTERPRETATION\ninterpretation here\n```\n where interpretation here is the actual interpretation to be transmitted and INTERPRETATION is just the word INTERPRETATION. Please provide an INTERPRETATION in a reasonable amount of time.\n"
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. INTERPRETATION, DIALOGUE).\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            phase_str = (
                "You are directing a PhD student to help them come up with a good plan, and you interact with them through dialogue.\n"
                "Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task, including what machine learning models to use and implement, what types of datasets should be searched for and used to train the model, and the exact details of the experiment.\n"
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are directing a PhD student to help them come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the postdoc your interpretation and use their feedback to improve your thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately. You must also complete this in a reasonable amount of time and then submit your results.\n"
            )
        return phase_str

    def role_description(self):
        return "a computer science postdoctoral student at a top university."

class MLEngineerAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
            "running experiments",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}")
        #elif phase == "running experiments":
        #    return (
        #        sr_str,
        #        f"Current Literature Review: {self.lit_review_sum}\n"
        #        f"Current Plan: {self.plan}\n"
        #        f"Current Dataset code: {self.dataset_code}\n"
        #    )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce code using the following command: ```python\ncode here\n```\n where code here is the actual code you will execute in a Python terminal, and python is just the word python. Try to incorporate some print functions. Do not use any classes or functions. If your code returns any errors, they will be provided to you, and you are also able to see print statements. You will receive all print statement results from the code. Make sure function variables are created inside the function or passed as a function parameter.\n"  # Try to avoid creating functions. 
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send, and DIALOGUE is just the word DIALOGUE.\n"
                "You also have access to HuggingFace datasets. You can search the datasets repository using the following command: ```SEARCH_HF\nsearch query here\n``` where search query here is the query used to search HuggingFace datasets, and SEARCH_HF is the word SEARCH_HF. This will return a list of HuggingFace dataset descriptions which can be loaded into Python using the datasets library. Your code MUST use an external HuggingFace directory.\n"
                "You MUST use a HuggingFace dataset in your code. DO NOT CREATE A MAIN FUNCTION. Try to make the code very simple.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. python, DIALOGUE, SEARCH_HF).\n")
        return ()

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            phase_str = (
                "You are a machine learning engineer being directed by a PhD student who will help you write the code, and you can interact with them through dialogue.\n"
                "Your goal is to produce code that prepares the data for the provided experiment. You should aim for simple code to prepare the data, not complex code. You should integrate the provided literature review and the plan and come up with code to prepare data for this experiment.\n"
            )
        return phase_str

    def role_description(self):
        return "a machine learning engineer working at a top university."

class SWEngineerAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}")
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the ML engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions. The submitted code must have a HuggingFace dataset import and must use an external HuggingFace dataset. If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Submit the code early. You should submit the code ASAP.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        elif phase == "data preparation":
            phase_str = (
                "You are a software engineer directing a machine learning engineer, where the machine learning engineer will be writing the code, and you can interact with them through dialogue.\n"
                "Your goal is to help the ML engineer produce code that prepares the data for the provided experiment. You should aim for very simple code to prepare the data, not complex code. You should integrate the provided literature review and the plan and come up with code to prepare data for this experiment.\n"
            )
        return phase_str

    def role_description(self):
        return "a software engineer working at a top university."
