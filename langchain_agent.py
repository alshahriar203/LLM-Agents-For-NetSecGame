import sys
from os import path
import argparse
import jinja2
import json

from dotenv import dotenv_values
from tenacity import retry, stop_after_attempt

# Set the logging
import logging

import numpy as np
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://147.32.83.60")
mlflow.set_experiment("LLM_QA")

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from env.game_components import ActionType, Action, IP, Data, Network, Service

# This is used so the agent can see the BaseAgent
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from base_agent import BaseAgent

# LangChain imports
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import re

config = dotenv_values(".env")

# Local services definition remains the same
local_services = ["can_attack_start_here"]

ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

# Define Pydantic models for structured output
class ActionParameters(BaseModel):
    target_network: Optional[str] = None
    target_host: Optional[str] = None
    target_service: Optional[str] = None
    data: Optional[Dict[str, str]] = None
    source_host: str

class LLMAction(BaseModel):
    action: str = Field(description="The action to take")
    parameters: ActionParameters

class ActionOutputParser(BaseOutputParser[LLMAction]):
    """Custom parser for LLM action output"""
    
    def parse(self, text: str) -> LLMAction:
        try:
            # Try to extract JSON from the text
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                return LLMAction(**data)
            else:
                # Try to parse the entire text as JSON
                data = json.loads(text.strip())
                return LLMAction(**data)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to construct from string
            lines = text.strip().split('\n')
            action_data = {}
            for line in lines:
                if 'action' in line.lower():
                    action_data['action'] = line.split(':')[1].strip().strip('"\'')
                elif 'parameters' in line.lower():
                    # Try to extract parameters
                    pass
            # Fallback to default action
            return LLMAction(
                action="ScanNetwork",
                parameters=ActionParameters(
                    source_host="0.0.0.0"
                )
            )

class PentesterAgent:
    """LangChain based pentester agent"""
    
    def __init__(self, llm_model: str, memory_buffer: int = 5):
        self.llm_model = llm_model
        self.memory_buffer = memory_buffer
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_len=memory_buffer * 3  # Approximate token count
        )
        
        # Initialize LLM based on model type
        if llm_model.startswith("gpt"):
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=0.0,
                api_key=config.get("OPENAI_API_KEY", "")
            )
            self.use_openai = True
        elif "zephyr" in llm_model or "mistral" in llm_model:
            # For local models, we'll use the original model_query function
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            self.model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.use_openai = False
        else:
            raise ValueError(f"Unsupported model: {llm_model}")
        
        # Create prompts
        self._create_prompts()
        
    def _create_prompts(self):
        """Create LangChain prompt templates"""
        
        # Chain of Thought examples
        self.cot_examples = """
Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24", "source_host": "2.2.2.2"}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3", "source_host": "2.2.2.2"}}
Action: {"action":"ExploitService", "parameters":{"target_host":"1.1.1.1", "target_service":"openssh", "source_host": "1.1.1.2"}}
Action: {"action":"FindData", "parameters":{"target_host":"1.1.1.1", "source_host": "1.1.1.2"}}
Action: {"action":"ExfiltrateData", "parameters": {"target_host": "2.2.2.2", "data": {"owner":"User1", "id":"WebData"}, "source_host": "1.1.1.2"}}
End of examples.
"""
        
        # Step 1: Object analysis prompt
        self.step1_template = PromptTemplate(
            input_variables=["instructions", "status"],
            template="""{instructions}

Current status:
{status}

List the objects in the current status and the actions they can be used. Be specific."""
        )
        
        # Step 2: Action selection prompt
        self.step2_template = PromptTemplate(
            input_variables=["instructions", "status", "cot_examples", "step1_response", "memory_prompt"],
            template="""{instructions}

Current status:
{status}

{cot_examples}

{step1_response}

{memory_prompt}

Provide the best next action in the correct JSON format. Action: """
        )
        
        # Create chains
        self.step1_chain = LLMChain(
            llm=self.llm if self.use_openai else None,
            prompt=self.step1_template,
            memory=self.memory
        )
        
        self.step2_chain = LLMChain(
            llm=self.llm if self.use_openai else None,
            prompt=self.step2_template,
            memory=self.memory,
            output_parser=ActionOutputParser()
        )
    
    def query_llm(self, messages: List[Dict[str, str]], max_tokens: int = 60) -> str:
        """Query the LLM (handles both OpenAI and local models)"""
        if self.use_openai:
            # Use LangChain's ChatOpenAI
            if messages[0]["role"] == "system":
                system_msg = messages[0]["content"]
                user_msg = "\n".join([m["content"] for m in messages[1:] if m["role"] == "user"])
                
                response = self.llm.predict(
                    f"{system_msg}\n\n{user_msg}",
                    max_tokens=max_tokens
                )
                return response
            else:
                # Convert to LangChain message format
                from langchain.schema import HumanMessage, SystemMessage
                
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                
                response = self.llm.invoke(langchain_messages)
                return response.content
        else:
            # Use local model with original model_query logic
            return self._model_query_local(messages, max_tokens)
    
    def _model_query_local(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Query local model using original logic"""
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        
        # Create chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generation config
        from transformers import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=True,
            eos_token_id=self.model.config.eos_token_id,
            temperature=0.1,
            top_k=100,
        )
        
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = self.model.generate(**model_inputs, generation_config=gen_config)
        return self.tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )[0]
    
    def add_memory(self, action: str, parameters: Dict[str, Any], feedback: str):
        """Add an action to memory"""
        memory_text = f"Action: {action} with parameters: {parameters}. {feedback}"
        self.memory.save_context(
            {"input": memory_text},
            {"output": ""}
        )


class LangChainPentester:
    """Main agent class refactored with LangChain"""
    
    def __init__(self, llm_agent: PentesterAgent, base_agent: BaseAgent):
        self.llm_agent = llm_agent
        self.base_agent = base_agent
        self.logger = logging.getLogger("llm_qa_langchain")
        
    @retry(stop=stop_after_attempt(3))
    def get_action(self, state, goal: str, memories: List) -> Optional[Action]:
        """Get next action using LangChain agent"""
        # Create status prompt
        status_prompt = self.create_status_from_state(state)
        
        # Create instructions
        jinja_environment = jinja2.Environment()
        template = jinja_environment.from_string("""
        You are a pentester and your goal is to {{goal}}.
        The actions you can take are ScanServices, FindData, ScanNetwork, ExploitService, and ExfiltrateData. 

        The rules are:
        1. You can only scan networks in your list of known networks.
        2. You can only scan services in hosts you know.
        3. You can only exploit services when you know the service.
        4. You can find data in hosts you control.
        5. You can exfiltrate known data to and from controlled hosts.

        Do not repeat actions that you took in the past.
        Do not scan or exploit services in hosts you control.
        Do not scan the same network twice.
        Do not exfiltrate the same data twice.
        Exploit services that you know and find data in all controlled hosts.
        Once you have known data exfiltrate them to the target one by one.
        """)
        instructions = template.render(goal=goal.lower())
        
        # Step 1: Object analysis
        messages = [
            {"role": "user", "content": instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": "List the objects in the current status and the actions they can be used. Be specific."}
        ]
        
        step1_response = self.llm_agent.query_llm(messages, max_tokens=1024)
        self.logger.info("LLM (step 1): %s", step1_response)
        
        # Step 2: Action selection
        memory_prompt = self.create_mem_prompt(memories[-self.llm_agent.memory_buffer:])
        
        messages = [
            {"role": "user", "content": instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": self.llm_agent.cot_examples},
            {"role": "user", "content": step1_response},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": "Provide the best next action in the correct JSON format. Action: "}
        ]
        
        response = self.llm_agent.query_llm(messages, max_tokens=80)
        self.logger.info("LLM (step 2): %s", response)
        
        # Parse response
        try:
            llm_action = self.llm_agent.step2_chain.output_parser.parse(response)
            is_valid, action = self.create_action_from_response(
                llm_action.dict(), state
            )
            return action if is_valid else None
        except Exception as e:
            self.logger.error(f"Cannot parse response from LLM: {response}, Error: {e}")
            return None
    
    def create_status_from_state(self, state):
        """Create a status prompt using the current state (same logic)"""
        contr_hosts = [host.ip for host in state.controlled_hosts]
        known_hosts = [
            str(host) for host in state.known_hosts if host.ip not in contr_hosts
        ]
        known_nets = [str(net) for net in list(state.known_networks)]

        prompt = "Current status:\n"
        prompt += f"Controlled hosts are {' and '.join(contr_hosts)}\n"
        self.logger.info("Controlled hosts are %s", " and ".join(contr_hosts))

        prompt += f"Known networks are {' and '.join(known_nets)}\n"
        self.logger.info("Known networks are %s", " and ".join(known_nets))
        prompt += f"Known hosts are {' and '.join(known_hosts)}\n"
        self.logger.info("Known hosts are %s", " and ".join(known_hosts))

        if len(state.known_services.keys()) == 0:
            prompt += "Known services are none\n"
            self.logger.info(f"Known services: None")
        for ip_service in state.known_services:
            services = []
            if len(list(state.known_services[ip_service])) > 0:
                for serv in state.known_services[ip_service]:
                    if serv.name not in local_services:
                        services.append(serv.name)
                if len(services) > 0:
                    serv_str = ""
                    for serv in services:
                        serv_str += serv + " and "
                    prompt += f"Known services for host {ip_service} are {serv_str}\n"
                    self.logger.info(f"Known services {ip_service, services}")
                else:
                    prompt += "Known services are none\n"
                    self.logger.info(f"Known services: None")

        if len(state.known_data.keys()) == 0:
            prompt += "Known data are none\n"
            self.logger.info(f"Known data: None")
        for ip_data in state.known_data:
            if len(state.known_data[ip_data]) > 0:
                host_data = ""
                for known_data in list(state.known_data[ip_data]):
                    host_data += f"({known_data.owner}, {known_data.id}) and "
                prompt += f"Known data for host {ip_data} are {host_data}\n"
                self.logger.info(f"Known data: {ip_data, state.known_data[ip_data]}")

        return prompt
    
    def create_mem_prompt(self, memory_list):
        """Summarize a list of memories into a few sentences (same logic)."""
        prompt = ""
        if len(memory_list) > 0:
            for memory in memory_list:
                prompt += f'You have taken action {{"action":"{memory[0]}" with "parameters":"{memory[1]}"}} in the past. {memory[2]}\n'
        return prompt
    
    def validate_action_in_state(self, llm_response, state):
        """Check the LLM response and validate it against the current state (same logic)."""
        contr_hosts = [str(host) for host in state.controlled_hosts]
        known_hosts = [
            str(host) for host in state.known_hosts if host.ip not in contr_hosts
        ]
        known_nets = [str(net) for net in list(state.known_networks)]

        valid = False
        try:
            action_str = llm_response["action"]
            action_params = llm_response["parameters"]
            if isinstance(action_params, str):
                action_params = eval(action_params)
            match action_str:
                case "ScanNetwork":
                    if action_params["target_network"] in known_nets:
                        valid = True
                case "ScanServices":
                    if (
                        action_params["target_host"] in known_hosts
                        or action_params["target_host"] in contr_hosts
                    ):
                        valid = True
                case "ExploitService":
                    ip_addr = action_params["target_host"]
                    if ip_addr in known_hosts:
                        valid = True
                case "FindData":
                    if action_params["target_host"] in contr_hosts:
                        valid = True
                case "ExfiltrateData":
                    for ip_data in state.known_data:
                        ip_addr = action_params["source_host"]
                        if ip_data == IP(ip_addr) and ip_addr in contr_hosts:
                            valid = True
                case _:
                    valid = False
            return valid
        except:
            self.logger.info("Exception during validation of %s", llm_response)
            return False
    
    def create_action_from_response(self, llm_response, state):
        """Build the action object from the llm response (same logic)"""
        try:
            valid = self.validate_action_in_state(llm_response, state)
            action = None
            action_str = llm_response["action"]
            action_params = llm_response["parameters"]
            if isinstance(action_params, str):
                action_params = eval(action_params)
            if valid:
                match action_str:
                    case "ScanNetwork":
                        target_net, mask = action_params["target_network"].split("/")
                        src_host = action_params["source_host"]
                        action = Action(
                            ActionType.ScanNetwork,
                            {
                                "target_network": Network(target_net, int(mask)),
                                "source_host": IP(src_host),
                            },
                        )
                    case "ScanServices":
                        src_host = action_params["source_host"]
                        action = Action(
                            ActionType.FindServices,
                            {
                                "target_host": IP(action_params["target_host"]),
                                "source_host": IP(src_host),
                            },
                        )
                    case "ExploitService":
                        target_ip = action_params["target_host"]
                        target_service = action_params["target_service"]
                        src_host = action_params["source_host"]
                        if len(list(state.known_services[IP(target_ip)])) > 0:
                            for serv in state.known_services[IP(target_ip)]:
                                if serv.name == target_service:
                                    parameters = {
                                        "target_host": IP(target_ip),
                                        "target_service": Service(
                                            serv.name,
                                            serv.type,
                                            serv.version,
                                            serv.is_local,
                                        ),
                                        "source_host": IP(src_host),
                                    }
                                    action = Action(ActionType.ExploitService, parameters)
                        else:
                            action = None
                    case "FindData":
                        src_host = action_params["source_host"]
                        action = Action(
                            ActionType.FindData,
                            {
                                "target_host": IP(action_params["target_host"]),
                                "source_host": IP(src_host),
                            },
                        )
                    case "ExfiltrateData":
                        try:
                            data_owner = action_params["data"]["owner"]
                            data_id = action_params["data"]["id"]
                        except:
                            action_data = eval(action_params["data"])
                            data_owner = action_data["owner"]
                            data_id = action_data["id"]

                        action = Action(
                            ActionType.ExfiltrateData,
                            {
                                "target_host": IP(action_params["target_host"]),
                                "data": Data(data_owner, data_id),
                                "source_host": IP(action_params["source_host"]),
                            },
                        )
                    case _:
                        return False, action

        except SyntaxError:
            self.logger.error(f"Cannot parse the response from the LLM: {llm_response}")
            valid = False

        return valid, action


def run_episode(langchain_agent, episode, num_iterations, goal, memory_buffer):
    """Run a single episode with LangChain agent"""
    actions_took_in_episode = []
    memories = []
    total_reward = 0
    repeated_actions = 0
    
    # Get initial observation
    observation = langchain_agent.base_agent.request_game_reset()
    current_state = observation.state
    
    for i in range(num_iterations):
        good_action = False
        
        # Get action from LangChain agent
        action = langchain_agent.get_action(observation.state, goal, memories)
        
        if action:
            # Execute action
            observation = langchain_agent.base_agent.make_step(action)
            total_reward += observation.reward
            
            if observation.state != current_state:
                good_action = True
                current_state = observation.state
                feedback = "This action was helpful."
            else:
                feedback = "This action was not helpful."
            
            # Add to memory
            memories.append((
                action.action_type.name,
                action.parameters,
                feedback
            ))
            
            # Track repeated actions
            if action in actions_took_in_episode:
                repeated_actions += 1
            actions_took_in_episode.append(action)
        else:
            # Invalid action
            memories.append((
                "Invalid",
                {},
                "This action was not valid based on your status."
            ))
        
        # Check if episode ended
        if observation.end or i == (num_iterations - 1):
            return {
                "steps": i,
                "total_reward": total_reward,
                "repeated_actions": repeated_actions,
                "reason": observation.info if observation.end else {"end_reason": "max_iterations"},
                "win": "goal_reached" in (observation.info.get("end_reason", "") if observation.end else "")
            }
    
    return {
        "steps": num_iterations,
        "total_reward": total_reward,
        "repeated_actions": repeated_actions,
        "reason": {"end_reason": "max_iterations"},
        "win": False
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        choices=[
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
        default="gpt-3.5-turbo",
        help="LLM used with OpenAI API",
    )
    parser.add_argument(
        "--test_episodes",
        help="Number of test episodes to run",
        default=30,
        action="store",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--memory_buffer",
        help="Number of actions to remember and pass to the LLM",
        default=5,
        action="store",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--host",
        help="Host where the game server is",
        default="127.0.0.1",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--port",
        help="Port where the game server is",
        default=9000,
        type=int,
        action="store",
        required=False,
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename="llm_qa_langchain.log",
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("llm_qa_langchain")
    logger.info("Start LangChain refactored agent")

    # Initialize base agent
    base_agent = BaseAgent(args.host, args.port, "Attacker")
    
    # Initialize LangChain agent
    llm_agent = PentesterAgent(args.llm, args.memory_buffer)
    langchain_agent = LangChainPentester(llm_agent, base_agent)

    # Setup MLFlow
    experiment_description = f"LLM QA agent with LangChain. Model: {args.llm}"
    mlflow.start_run(description=experiment_description)

    params = {
        "model": args.llm,
        "memory_len": args.memory_buffer,
        "episodes": args.test_episodes,
        "framework": "LangChain"
    }
    mlflow.log_params(params)

    # Run episodes
    wins = 0
    detected = 0
    reach_max_steps = 0
    returns = []
    num_steps = []
    num_win_steps = []
    num_detected_steps = []
    num_actions_repeated = []
    
    # Get initial observation to get goal
    observation = base_agent.register()
    
    for episode in range(1, args.test_episodes + 1):
        logger.info(f"Running episode {episode}")
        print(f"Running episode {episode}")
        
        # Reset game
        observation = base_agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        goal = observation.info["goal_description"]
        
        # Run episode
        result = run_episode(
            langchain_agent, 
            episode, 
            num_iterations, 
            goal, 
            args.memory_buffer
        )
        
        # Process results
        if result["win"]:
            wins += 1
            num_win_steps.append(result["steps"])
            logger.info(f"Episode {episode}: Win after {result['steps']} steps")
        elif "detected" in result["reason"].get("end_reason", ""):
            detected += 1
            num_detected_steps.append(result["steps"])
            logger.info(f"Episode {episode}: Detected after {result['steps']} steps")
        else:
            reach_max_steps += 1
            logger.info(f"Episode {episode}: Max steps reached")
        
        returns.append(result["total_reward"])
        num_steps.append(result["steps"])
        num_actions_repeated.append(result["repeated_actions"])
        
        # Log metrics to MLFlow
        mlflow.log_metric("wins", wins, step=episode)
        mlflow.log_metric("num_steps", result["steps"], step=episode)
        mlflow.log_metric("return", result["total_reward"], step=episode)
        mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
        mlflow.log_metric("detected", detected, step=episode)
        mlflow.log_metric("win_rate", (wins / episode) * 100, step=episode)
        mlflow.log_metric("avg_returns", np.mean(returns), step=episode)
        mlflow.log_metric("avg_steps", np.mean(num_steps), step=episode)
    
    # Calculate final statistics
    test_win_rate = (wins / args.test_episodes) * 100
    test_detection_rate = (detected / args.test_episodes) * 100
    test_max_steps_rate = (reach_max_steps / args.test_episodes) * 100
    test_average_returns = np.mean(returns)
    test_std_returns = np.std(returns)
    test_average_episode_steps = np.mean(num_steps)
    test_std_episode_steps = np.std(num_steps)
    
    # Log final metrics
    tensorboard_dict = {
        "test_avg_win_rate": test_win_rate,
        "test_avg_detection_rate": test_detection_rate,
        "test_avg_max_steps_rate": test_max_steps_rate,
        "test_avg_returns": test_average_returns,
        "test_std_returns": test_std_returns,
        "test_avg_episode_steps": test_average_episode_steps,
        "test_std_episode_steps": test_std_episode_steps,
    }
    
    if num_win_steps:
        tensorboard_dict["test_avg_win_steps"] = np.mean(num_win_steps)
        tensorboard_dict["test_std_win_steps"] = np.std(num_win_steps)
    
    if num_detected_steps:
        tensorboard_dict["test_avg_detected_steps"] = np.mean(num_detected_steps)
        tensorboard_dict["test_std_detected_steps"] = np.std(num_detected_steps)
    
    tensorboard_dict["test_avg_repeated_steps"] = np.mean(num_actions_repeated)
    tensorboard_dict["test_std_repeated_steps"] = np.std(num_actions_repeated)
    
    mlflow.log_metrics(tensorboard_dict)
    
    # Print summary
    text = f"""Final test after {args.test_episodes} episodes
        Wins={wins},
        Detections={detected},
        winrate={test_win_rate:.3f}%,
        detection_rate={test_detection_rate:.3f}%,
        max_steps_rate={test_max_steps_rate:.3f}%,
        average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
        average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f}"""
    
    if num_win_steps:
        text += f",\n        average_win_steps={np.mean(num_win_steps):.3f} +- {np.std(num_win_steps):.3f}"
    if num_detected_steps:
        text += f",\n        average_detected_steps={np.mean(num_detected_steps):.3f} +- {np.std(num_detected_steps):.3f}"
    
    text += f",\n        average_repeated_steps={np.mean(num_actions_repeated):.3f} +- {np.std(num_actions_repeated):.3f}"
    
    print(text)
    logger.info(text)
    mlflow.end_run()