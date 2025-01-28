from agents import *
from copy import copy
from common_imports import *
from mlesolver import MLESolver
import argparse
import pickle

DEFAULT_LLM_BACKBONE = "o1-mini"

class LaboratoryWorkflow:
    def __init__(self, research_topic, openai_api_key, max_steps=100, num_papers_lit_review=5, agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), 
             human_in_loop_flag={"literature review": False, "plan formulation": False, 
                                "data preparation": False, "running experiments": False,
                                "results interpretation": False, "report writing": False,
                                "report refinement": False}, 
             compile_pdf=True, mlesolver_max_steps=3, papersolver_max_steps=5):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """

        self.notes = notes
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review
        
        # Configure LM Studio if needed
        if isinstance(agent_model_backbone, str) and agent_model_backbone.startswith("lmstudio-"):
            os.environ['OPENAI_API_BASE'] = "http://localhost:1234/v1"
            print("\n🔧 Configuring LM Studio integration...")
            try:
                client = OpenAI(
                    base_url="http://localhost:1234/v1",
                    api_key="lm-studio"
                )
                models = client.models.list()
                model_name = agent_model_backbone.replace('lmstudio-', '')
                available_models = [model.id for model in models.data]
                if model_name in available_models:
                    print("✅ LM Studio model verified and ready")
                else:
                    print(f"⚠️ Warning: Model {model_name} not found in LM Studio")
                    print(f"Available models: {available_models}")
            except Exception as e:
                print(f"⚠️ Warning: Could not verify LM Studio connection: {e}")
                print("Please ensure LM Studio server is running on port 1234")
        
        self.print_cost = True
        self.review_override = True
        self.review_ovrd_steps = 0
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        self.num_ref_papers = 1
        self.review_total_steps = 1
        self.arxiv_num_summaries = 5
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            ("results interpretation", ["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif type(agent_model_backbone) == dict:
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0,},
            "plan formulation":       {"time": 0.0, "steps": 0.0,},
            "data preparation":       {"time": 0.0, "steps": 0.0,},
            "running experiments":    {"time": 0.0, "steps": 0.0,},
            "results interpretation": {"time": 0.0, "steps": 0.0,},
            "report writing":         {"time": 0.0, "steps": 0.0,},
            "report refinement":      {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.phd = PhDStudentAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.postdoc = PostdocAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.professor = ProfessorAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.ml_engineer = MLEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.sw_engineer = SWEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)

        # remove previous files
        remove_figures()
        remove_directory("research_dir")
        # make src and research directory
        if not os.path.exists("state_saves"):
            os.mkdir(os.path.join(".", "state_saves"))
        os.mkdir(os.path.join(".", "research_dir"))
        os.mkdir(os.path.join("./research_dir", "src"))
        os.mkdir(os.path.join("./research_dir", "tex"))

    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        phase = phase.replace(" ", "_")
        with open(f"state_saves/{phase}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose: print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            for subtask in subtasks:
                if self.verbose: print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else: self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    repeat = True
                    while repeat: repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: self.save_state(subtask)
                        return

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save: self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            response = input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
            return response.lower().strip()[0] == 'y'
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.phd.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic, phase="report refinement", feedback=review_prompt, step=0)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if self.verbose:  # Changed from verbose to self.verbose
                    print("*"*40, "\n", "REVIEW COMPLETE", "\n", "*"*40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else: 
                raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [_note["note"] for _note in self.ml_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        # instantiate mle-solver
        from papersolver import PaperSolver
        self.reference_papers = []
        solver = PaperSolver(
            notes=report_notes, 
            max_steps=self.papersolver_max_steps, 
            plan=self.phd.plan,  # Changed from lab.phd.plan
            exp_code=self.phd.results_code,  # Changed from lab.phd.results_code
            exp_results=self.phd.exp_results,  # Changed from lab.phd.exp_results
            insights=self.phd.interpretation,  # Changed from lab.phd.interpretation
            lit_review=self.phd.lit_review,  # Changed from lab.phd.lit_review
            ref_papers=self.reference_papers,
            topic=self.research_topic,  # Changed from research_topic
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["report writing"] if isinstance(self.model_backbone, dict) else self.model_backbone,
            compile_pdf=self.compile_pdf  # Changed from compile_pdf
        )
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps):
            solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        if self.verbose: print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry: return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file("./research_dir", "readme.md", readme)
        save_to_file("./research_dir", "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            resp = self.postdoc.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry: return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        experiment_notes = [_note["note"] for _note in self.ml_engineer.notes if "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        # instantiate mle-solver
        llm_str = self.model_backbone["running experiments"] if isinstance(self.model_backbone, dict) else self.model_backbone
        solver = MLESolver(dataset_code=self.ml_engineer.dataset_code, 
                          notes=experiment_notes, 
                          insights=self.ml_engineer.lit_review_sum, 
                          max_steps=self.mlesolver_max_steps, 
                          plan=self.ml_engineer.plan, 
                          openai_api_key=self.openai_api_key, 
                          llm_str=llm_str)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps-1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # regenerate figures from top code
        execute_code(code)
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose: print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("running experiments", code)  # Corregido de "data preparation"
            if retry: return retry
        save_to_file("./research_dir/src", "run_experiments.py", code)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        # reset agent state
        self.reset_agents()
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            if ml_feedback != "":
                ml_feedback_in = "Feedback provided to the ML agent: " + ml_feedback
            else: ml_feedback_in = ""
            resp = self.sw_engineer.inference(self.research_topic, "data preparation", feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}", step=_i)
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose: print("#"*40, f"\nThe following is dialogue produced by the SW Engineer: {dialogue}", "\n", "#"*40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry: return retry
                    save_to_file("./research_dir/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            if ml_feedback != "":
                ml_feedback_in = "Feedback from previous command: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.ml_engineer.inference(
                self.research_topic, "data preparation",
                feedback=f"{swe_dialogue}\n{ml_feedback_in}", step=_i)
            #if self.verbose: print("ML Engineer: ", resp, "\n~~~~~~~~~~~")
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose: print("#" * 40, f"\nThe following is dialogue produced by the ML Engineer: {dialogue}", "#" * 40, "\n")
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            # inference postdoc to
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry: return retry
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Plan Formulation")

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        arx_eng = ArxivSearch()
        max_tries = self.max_steps * 5  # lit review often requires extra steps
        try:
            # get initial response from PhD agent with timeout
            resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.8)
            if resp is None or resp.strip() == "":
                print("Warning: Empty response from PhD agent, retrying phase...")
                return True

            if self.verbose:
                print("\n" + "="*80)
                print("📚 Literature Review Phase".center(80))
                print("="*80 + "\n")
                print("Initial PhD response:", resp, "\n" + "-"*40)

            for _i in range(max_tries):
                feedback = str()
                
                if _i > 0 and _i % 10 == 0:
                    print(f"\n🔄 Progress: Step {_i}/{max_tries}")

                if "```SUMMARY" in resp:
                    query = extract_prompt(resp, "SUMMARY")
                    print(f"\n🔍 Searching papers for: {query}")
                    papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                    feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

                elif "```FULL_TEXT" in resp:
                    query = extract_prompt(resp, "FULL_TEXT")
                    print(f"\n📄 Retrieving full text for paper: {query}")
                    arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + arx_eng.retrieve_full_paper_text(query) + "```"
                    feedback = arxiv_paper

                elif "```ADD_PAPER" in resp:
                    query = extract_prompt(resp, "ADD_PAPER")
                    print(f"\n➕ Adding paper to review: {query}")
                    feedback, text = self.phd.add_review(query, arx_eng)
                    if len(self.reference_papers) < self.num_ref_papers:
                        self.reference_papers.append(text)

                if len(self.phd.lit_review) >= self.num_papers_lit_review:
                    lit_review_sum = self.phd.format_review()
                    if lit_review_sum and lit_review_sum.strip():
                        if self.human_in_loop_flag["literature review"]:
                            print("\n" + "="*80)
                            print("📋 Literature Review Summary".center(80))
                            print("="*80 + "\n")
                            print(lit_review_sum)
                            print("\n" + "="*80)
                            retry = self.human_in_loop("literature review", lit_review_sum)
                            if retry:
                                self.phd.lit_review = []
                                return retry
                        if self.verbose:
                            print(self.phd.lit_review_sum)
                        self.set_agent_attr("lit_review_sum", lit_review_sum)
                        self.reset_agents()
                        self.statistics_per_phase["literature review"]["steps"] = _i
                        return False
                
                resp = self.phd.inference(self.research_topic, "literature review", 
                                        feedback=feedback, step=_i + 1, temp=0.8)
                if resp is None or resp.strip() == "":
                    print("\n⚠️ Warning: Empty response from PhD agent, retrying step...")
                    continue

        except Exception as e:
            print(f"\n❌ Error in literature review: {str(e)}")
            if "Max retries exceeded" in str(e):
                print("⏳ ArXiv API rate limit reached. Waiting before retrying...")
                time.sleep(60)
            return True

        raise Exception("Max tries during phase: Literature Review")

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y": pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else: print("Invalid response, type Y or N")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")

    parser.add_argument(
        '--copilot-mode',
        type=str,
        default="False",
        help='Enable human interaction mode.'
    )

    parser.add_argument(
        '--deepseek-api-key',
        type=str,
        help='Provide the DeepSeek API key.'
    )

    parser.add_argument(
        '--load-existing',
        type=str,
        default="False",
        help='Do not load existing state; start a new workflow.'
    )

    parser.add_argument(
        '--load-existing-path',
        type=str,
        help='Path to load existing state; start a new workflow, e.g. state_saves/results_interpretation.pkl'
    )

    parser.add_argument(
        '--research-topic',
        type=str,
        help='Specify the research topic.'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Provide the OpenAI API key.'
    )

    parser.add_argument(
        '--compile-latex',
        type=str,
        default="True",
        help='Compile latex into pdf during paper writing phase. Disable if you can not install pdflatex.'
    )

    parser.add_argument(
        '--llm-backend',
        type=str,
        default="o1-mini",
        help='Backend LLM to use for agents in Agent Laboratory.'
    )

    parser.add_argument(
        '--language',
        type=str,
        default="English",
        help='Language to operate Agent Laboratory in.'
    )

    parser.add_argument(
        '--num-papers-lit-review',
        type=int,  # Cambiar de str a int
        default=5,
        help='Total number of papers to summarize in literature review stage'
    )

    parser.add_argument(
        '--mlesolver-max-steps',
        type=int,  # Cambiar de str a int
        default=3,
        help='Total number of mle-solver steps'
    )

    parser.add_argument(
        '--papersolver-max-steps',
        type=int,  # Cambiar de str a int
        default=5,
        help='Total number of paper-solver steps'
    )

    parser.add_argument(
        '--ollama-model',
        type=str,
        help='Specify the Ollama model to use (e.g., llama2, codellama, deepseek-r1)'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    llm_backend = args.llm_backend
    human_mode = args.copilot_mode.lower() == "true"
    compile_pdf = args.compile_latex.lower() == "true"
    load_existing = args.load_existing.lower() == "true"
    
    # Fix: Remove .lower() since args.num_papers_lit_review is already an int
    num_papers_lit_review = args.num_papers_lit_review
    papersolver_max_steps = args.papersolver_max_steps
    mlesolver_max_steps = args.mlesolver_max_steps

    # Modify the API key validation logic
    if args.llm_backend.startswith("ollama-"):
        api_key = None  # No API key needed for Ollama
    elif args.llm_backend.startswith("lmstudio-"):
        api_key = "dummy_key"  # Use dummy key for LM Studio
        os.environ['OPENAI_API_BASE'] = "http://localhost:1234/v1"
        print(f"Using LM Studio with model: {args.llm_backend.replace('lmstudio-', '')}")
    elif args.api_key:
        api_key = args.api_key
    elif args.deepseek_api_key and args.llm_backend in ["deepseek-chat", "deepseek-reasoner"]:
        api_key = args.deepseek_api_key
    elif os.getenv('OPENAI_API_KEY'):
        api_key = os.getenv('OPENAI_API_KEY')
    elif os.getenv('DEEPSEEK_API_KEY') and args.llm_backend in ["deepseek-chat", "deepseek-reasoner"]:
        api_key = os.getenv('DEEPSEEK_API_KEY')
    else:
        if not args.llm_backend.startswith("ollama-"):  # Only raise error if not using Ollama
            raise ValueError("API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")

    ##########################################################
    # Research question that the agents are going to explore #
    ##########################################################
    if human_mode or args.research_topic is None:
        research_topic = input("Please name an experiment idea for AgentLaboratory to perform: ")
    else:
        research_topic = args.research_topic

    task_notes_LLM = [
        {"phases": ["plan formulation"],
         "note": "You should come up with a plan for TWO experiments."},

        {"phases": ["plan formulation", "data preparation", "running experiments"],
         "note": "Please use local models through Ollama for your experiments."},

        {"phases": ["running experiments"],
         "note": "Since we're using local models through Ollama, there's no need to worry about API costs or rate limits. Feel free to make multiple model calls as needed for better results."},

        {"phases": ["running experiments"],
         "note": "Use a reasonable dataset size (up to 1000 data points) since we're running inference locally and don't have API constraints."},

        {"phases": ["data preparation", "running experiments"],
         "note": "For GPU acceleration: Use 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' for systems without GPU support. The code should automatically detect and use the best available device."},

        {"phases": ["data preparation", "running experiments"],
         "note": "Generate figures with very colorful and artistic design."},

        {"phases": ["running experiments", "results interpretation"],
         "note": "Take advantage of the local model's speed to perform multiple iterations and validations of your results."}
    ]

    task_notes_LLM.append(
        {"phases": ["literature review", "plan formulation", "data preparation", "running experiments", "results interpretation", "report writing", "report refinement"],
        "note": f"You should always write in the following language to converse and to write the report {args.language}"},
    )

    ####################################################
    ###  Stages where human input will be requested  ###
    ####################################################
    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    ###################################################
    ###  LLM Backend used for the different phases  ###
    ###################################################
    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "paper refinement":       llm_backend,
    }

    if load_existing:
        load_path = args.load_existing_path
        if load_path is None:
            raise ValueError("Please provide path to load existing state.")
        with open(load_path, "rb") as f:
            lab = pickle.load(f)
    else:
        lab = LaboratoryWorkflow(
            research_topic=research_topic,
            notes=task_notes_LLM,
            agent_model_backbone=agent_models,
            human_in_loop_flag=human_in_loop,
            openai_api_key=api_key,
            compile_pdf=compile_pdf,
            num_papers_lit_review=num_papers_lit_review,
            papersolver_max_steps=papersolver_max_steps,
            mlesolver_max_steps=mlesolver_max_steps,
        )

    lab.perform_research()
