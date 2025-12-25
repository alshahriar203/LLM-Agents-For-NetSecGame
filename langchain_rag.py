import pandas as pd
import numpy as np
import time
import json
import argparse
import logging
import sys
from os import path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import jinja2

import chromadb
from chromadb.utils import embedding_functions
from dotenv import dotenv_values

import mlflow
mlflow.set_experiment("mari_LLM_RAG_test")

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain

# Local imports
from rag_utils import select_best_action
from env.game_components import ActionType
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from base_agent import BaseAgent
from llm_utils import (
    create_status_from_state,
    create_action_from_response,
)

# Constants
local_services = ["can_attack_start_here"]

ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}

COT_PROMPT2 = """
Here are some examples of actions:
Action: {"action":"ScanNetwork", "parameters": {"target_network": "1.1.1.0/24", 'source_host': '2.2.2.2'}}
Action: {"action":"ScanServices", "parameters":{"target_host":"2.2.2.3"}, 'source_host': '2.2.2.2'}}
Action: {"action":"ExploitService", "parameters":{"target_host":"1.1.1.1", "target_service":"openssh", 'source_host': '1.1.1.2'}}
Action: {"action":"FindData", "parameters":{"target_host":"1.1.1.1", 'source_host': '1.1.1.2'}}
Action: {"action":"ExfiltrateData", "parameters": {"target_host": "2.2.2.2", "data": {"owner":"User1", "id":"WebData"}, "source_host": "1.1.1.2"}}
End of examples.
"""

INSTRUCTIONS_TEMPLATE = """
You are a pentester and your goal is to {goal}.
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

{past_actions}

Current status:
{status}

Examples:
{examples}

Provide the best next {selected_action} action in the correct JSON format if it is valid based on the provided rules.
Otherwise, provide the best possible action in the correct JSON format.
Action: """

@dataclass
class AgentConfig:
    """Configuration for the RAG agent"""
    llm_model: str
    memory_buffer: int
    test_episodes: int
    host: str
    port: int
    embeddings_model: str
    database_folder: str

class RAGAgentLangChain:
    """Simplified LangChain-only RAG Agent"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.llm = self._setup_llm()
        self.vector_store = self._setup_vector_store()
        self.prompt_template = self._setup_prompt_template()
        
    def _setup_logger(self):
        """Setup logger"""
        logging.basicConfig(
            filename="llm_rag_langchain.log",
            filemode="w",
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
        return logging.getLogger("llm_rag_langchain")
    
    def _setup_llm(self):
        """Setup language model"""
        config = dotenv_values(".env")
        
        if self.config.llm_model in ["llama3", "zephyr", "netsec4bit", "netsec_full", "codellama"]:
            return ChatOpenAI(
                base_url="http://147.32.83.61:11434/v1",
                api_key="ollama",
                model=self.config.llm_model,
                temperature=0.0,
                max_tokens=80,
            )
        else:
            return ChatOpenAI(
                api_key=config["OPENAI_API_KEY"],
                model=self.config.llm_model,
                temperature=0.0,
                max_tokens=80,
            )
    
    def _setup_vector_store(self):
        """Setup vector store for RAG"""
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embeddings_model
        )
        
        db_client = chromadb.PersistentClient(path=self.config.database_folder)
        collection = db_client.get_collection("states")
        
        return collection
    
    def _setup_prompt_template(self):
        """Create LangChain prompt template"""
        return PromptTemplate(
            input_variables=["goal", "past_actions", "status", "examples", "selected_action"],
            template=INSTRUCTIONS_TEMPLATE,
        )
    
    def create_mem_prompt(self, memory_list: List[Any]) -> str:
        """Summarize memories into prompt"""
        prompt = ""
        if len(memory_list) > 0:
            for memory in memory_list:
                prompt += f'You have taken action {{"action":"{memory[0]}" with "parameters":"{memory[1]}"}} in the past. {memory[2]}\n'
        return prompt
    
    def query_vector_store(self, status_prompt: str) -> str:
        """Query vector store for similar states"""
        state_embedding = self.vector_store._embedding_function([status_prompt])
        results = self.vector_store.query(
            query_embeddings=state_embedding,
            n_results=5,
        )
        selected_action = select_best_action(results["metadatas"][0])
        self.logger.info(f"The selected action is: {selected_action}")
        return selected_action
    
    def create_llm_chain(self):
        """Create LangChain LLM chain"""
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=False,
        )
    
    def run_episode(self, agent: BaseAgent, episode_num: int) -> Dict[str, Any]:
        """Run a single episode using LangChain"""
        # Reset the game
        observation = agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        goal = observation.info["goal_description"]
        current_state = observation.state
        
        memories = []
        actions_took_in_episode = []
        good_states = []
        good_metadata = []
        
        total_reward = 0
        repeated_actions = 0
        store_data = False
        
        llm_chain = self.create_llm_chain()
        
        for step in range(num_iterations):
            # Step 1: Create status prompt
            status_prompt = create_status_from_state(observation.state)
            
            # Step 2: Query vector store
            selected_action = self.query_vector_store(status_prompt)
            
            # Step 3: Build prompt with memory
            past_actions = self.create_mem_prompt(memories[-self.config.memory_buffer:])
            
            # Step 4: Call LLM through LangChain
            response = llm_chain.run(
                goal=goal.lower(),
                past_actions=past_actions,
                status=status_prompt,
                examples=COT_PROMPT2,
                selected_action=selected_action
            )
            
            self.logger.info(f"LLM response: {response.strip()}")
            
            # Step 5: Parse and validate response
            try:
                response_dict = json.loads(response.strip())
                is_valid, action = create_action_from_response(
                    response_dict, observation.state
                )
            except:
                self.logger.error("Failed to parse LLM response")
                is_valid = False
                action = None
            
            # Step 6: Execute action
            if is_valid and action is not None:
                observation = agent.make_step(action)
                total_reward += observation.reward
                
                # Check if action changed state
                if observation.state != current_state:
                    good_action = True
                    current_state = observation.state
                    
                    # Store good state and metadata
                    action_str = action.type.name
                    if action_str == "FindServices":
                        action_str = "ScanServices"
                    
                    good_states.append(status_prompt)
                    good_metadata.append({"action": action_str})
                    
                    memory_text = "This action was helpful."
                else:
                    good_action = False
                    memory_text = "This action was not helpful."
                
                # Update memory
                memories.append((
                    response_dict["action"],
                    response_dict["parameters"],
                    memory_text
                ))
                
                # Check for repeated actions
                if action in actions_took_in_episode:
                    repeated_actions += 1
                actions_took_in_episode.append(action)
                
            else:
                # Invalid action
                if 'response_dict' in locals():
                    memories.append((
                        response_dict.get("action", "Invalid"),
                        response_dict.get("parameters", {}),
                        "This action was not valid based on your status."
                    ))
                else:
                    memories.append((
                        "Invalid",
                        {},
                        "Response was badly formatted."
                    ))
            
            # Check termination
            if observation.end or step == (num_iterations - 1):
                if step < (num_iterations - 1):
                    reason = observation.info
                else:
                    reason = {"end_reason": "max_iterations"}
                
                # Determine end type
                if "goal_reached" in reason.get("end_reason", ""):
                    type_of_end = "win"
                    store_data = True
                elif "detected" in reason.get("end_reason", ""):
                    type_of_end = "detection"
                elif "max_iterations" in reason.get("end_reason", ""):
                    type_of_end = "max_iterations"
                else:
                    type_of_end = "max_steps"
                
                break
        
        # Store data if successful episode
        if store_data and good_states:
            self._store_in_vector_db(good_states, good_metadata)
        
        return {
            "episode": episode_num,
            "steps": step + 1,
            "total_reward": total_reward,
            "type_of_end": type_of_end,
            "repeated_actions": repeated_actions,
        }
    
    def _store_in_vector_db(self, good_states: List[str], good_metadata: List[Dict]):
        """Store successful state-action pairs in vector database"""
        num_docs = self.vector_store.count()
        good_embeddings = self.vector_store._embedding_function(good_states)
        
        self.vector_store.add(
            embeddings=good_embeddings,
            metadatas=good_metadata,
            documents=[f"doc{i}" for i in range(num_docs, num_docs + len(good_metadata))],
            ids=[f"state{i}" for i in range(num_docs, num_docs + len(good_metadata))]
        )
        self.logger.info(f"Stored {len(good_metadata)} new embeddings")

def run_experiment_simple(config: AgentConfig):
    """Run the experiment using simplified LangChain approach"""
    run_name = f"netsecgame__llm_rag_langchain__{int(time.time())}"
    experiment_description = f"LLM RAG agent (LangChain). Model: {config.llm_model}"
    
    with mlflow.start_run(description=experiment_description) as run:
        # Log parameters
        params = {
            "model": config.llm_model,
            "memory_len": config.memory_buffer,
            "episodes": config.test_episodes,
        }
        mlflow.log_params(params)
        
        # Initialize agent
        agent = BaseAgent(config.host, config.port, "Attacker")
        rag_agent = RAGAgentLangChain(config)
        
        # Run episodes
        wins = 0
        detected = 0
        reach_max_steps = 0
        returns = []
        num_steps = []
        num_win_steps = []
        num_detected_steps = []
        num_actions_repeated = []
        
        for episode in range(1, config.test_episodes + 1):
            rag_agent.logger.info(f"Running episode {episode}")
            print(f"Running episode {episode}")
            
            # Run episode
            episode_result = rag_agent.run_episode(agent, episode)
            
            # Update statistics
            type_of_end = episode_result["type_of_end"]
            steps = episode_result["steps"]
            total_reward = episode_result["total_reward"]
            
            if type_of_end == "win":
                wins += 1
                num_win_steps.append(steps)
            elif type_of_end == "detection":
                detected += 1
                num_detected_steps.append(steps)
            elif type_of_end in ["max_iterations", "max_steps"]:
                reach_max_steps += 1
            
            returns.append(total_reward)
            num_steps.append(steps)
            num_actions_repeated.append(episode_result["repeated_actions"])
            
            # Log metrics
            mlflow.log_metric("wins", wins, step=episode)
            mlflow.log_metric("num_steps", steps, step=episode)
            mlflow.log_metric("return", total_reward, step=episode)
            mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
            mlflow.log_metric("detected", detected, step=episode)
            mlflow.log_metric("win_rate", (wins / episode) * 100, step=episode)
            mlflow.log_metric("avg_returns", np.mean(returns), step=episode)
            mlflow.log_metric("avg_steps", np.mean(num_steps), step=episode)
            
            rag_agent.logger.info(
                f"Episode {episode} ended after {steps} steps. Type: {type_of_end}"
            )
            print(f"Episode {episode} ended after {steps} steps. Type: {type_of_end}")
        
        # Calculate final statistics
        test_win_rate = (wins / config.test_episodes) * 100
        test_detection_rate = (detected / config.test_episodes) * 100
        test_max_steps_rate = (reach_max_steps / config.test_episodes) * 100
        test_average_returns = np.mean(returns)
        test_std_returns = np.std(returns)
        test_average_episode_steps = np.mean(num_steps)
        test_std_episode_steps = np.std(num_steps)
        test_average_win_steps = np.mean(num_win_steps)
        test_std_win_steps = np.std(num_win_steps)
        test_average_detected_steps = np.mean(num_detected_steps)
        test_std_detected_steps = np.std(num_detected_steps)
        test_average_repeated_steps = np.mean(num_actions_repeated)
        test_std_repeated_steps = np.std(num_actions_repeated)
        
        # Log final metrics
        tensorboard_dict = {
            "test_avg_win_rate": test_win_rate,
            "test_avg_detection_rate": test_detection_rate,
            "test_avg_max_steps_rate": test_max_steps_rate,
            "test_avg_returns": test_average_returns,
            "test_std_returns": test_std_returns,
            "test_avg_episode_steps": test_average_episode_steps,
            "test_std_episode_steps": test_std_episode_steps,
            "test_avg_win_steps": test_average_win_steps,
            "test_std_win_steps": test_std_win_steps,
            "test_avg_detected_steps": test_average_detected_steps,
            "test_std_detected_steps": test_std_detected_steps,
            "test_avg_repeated_steps": test_average_repeated_steps,
            "test_std_repeated_steps": test_std_repeated_steps,
        }
        
        mlflow.log_metrics(tensorboard_dict)
        
        # Print final results
        text = f"""Final test after {config.test_episodes} episodes
        Wins={wins},
        Detections={detected},
        winrate={test_win_rate:.3f}%,
        detection_rate={test_detection_rate:.3f}%,
        max_steps_rate={test_max_steps_rate:.3f}%,
        average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
        average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
        average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
        average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
        average_repeated_steps={test_average_repeated_steps:.3f} +- {test_std_repeated_steps:.3f}"""
        
        print(text)
        rag_agent.logger.info(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        choices=[
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "llama3",
            "zephyr",
            "netsec4bit",
            "netsec_full",
            "codellama",
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
    parser.add_argument(
        "--embeddings_model",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="LLM used to create embeddings",
    )
    parser.add_argument("--database_folder", type=str, default="embeddings_db")
    args = parser.parse_args()
    
    # Create configuration
    config = AgentConfig(
        llm_model=args.llm,
        memory_buffer=args.memory_buffer,
        test_episodes=args.test_episodes,
        host=args.host,
        port=args.port,
        embeddings_model=args.embeddings_model,
        database_folder=args.database_folder
    )
    
    # Run experiment
    run_experiment_simple(config)