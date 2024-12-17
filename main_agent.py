import os
import uuid
from typing import Annotated
from fastapi import FastAPI, APIRouter
from typing_extensions import TypedDict
from IPython.display import Image,display
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage


class AgentState(TypedDict):
    input: str
    category: str
    messages: Annotated[list, add_messages]

class MainAgent:
    def __init__(self):
        self.LLM = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                                          api_key=os.getenv('API_KEY'))
        self.graph_builder = StateGraph(AgentState)
        self.memory = MemorySaver()
        self.__build_graph()

    def __build_graph(self):
        """Build the unified state graph."""
        # Add nodes
        self.graph_builder.add_node("greeting", self.greet_user)
        self.graph_builder.add_node("enter_category", self.enter_category)
        self.graph_builder.add_node("pass_by", self.pass_by)
        self.graph_builder.add_node("list_products", self.list_products)
        self.graph_builder.add_node("category_not_found", self.category_not_found)
        self.graph_builder.add_node("unknown_handler", self.handle_unknown_messages)

        # Add edges
        self.graph_builder.add_edge(START, "greeting")
        self.graph_builder.add_edge("greeting", "enter_category")
        self.graph_builder.add_conditional_edges("enter_category", self.coordinator, {
            'category': "pass_by",
            'unknown': "unknown_handler"
        })
        self.graph_builder.add_conditional_edges("pass_by", self.check_category, {
            'true': "list_products",
            'false': "category_not_found"
        })
        self.graph_builder.add_edge("list_products", "pass_by")
        self.graph_builder.add_edge("category_not_found", "pass_by")
        self.graph_builder.add_edge("unknown_handler", "enter_category")

        # Compile the graph
        self.graph = self.graph_builder.compile(
            interrupt_after=['enter_category','unknown_handler','category_not_found','list_products'],
            checkpointer=self.memory
        )

    def greet_user(self, state: AgentState):
        """Greets the user with a welcome message."""
        return {"messages": "Hello! Welcome to our e-commerce store."}
    
    def end(self,state:AgentState):
        return {"messages": ""}
    
    def enter_category(self, state: AgentState):
        """Handles user category input."""
        return {"messages": "Please enter the category you are interested in."}

    def coordinator(self, state: AgentState):
        """Categorizes the user input as category-related or unknown."""
        message = state["messages"][-1]
        print(f"Last Message: {message}")

        prompt = (
            "You are a shop coordinator responsible for directing the user to the right place.\n"
            "Check if the user is asking about a certain category or something general related to products or categories.\n"
            "Answer with 'c' for category or 'u' for unknown."
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt), ("human", "Question: {question}")
        ])
        result = (prompt_template | self.LLM).invoke(message)
        return 'category'  if 'c' in result.content  else 'unknown'

    def pass_by(self, state: AgentState):
        return {'messages':state['messages'][-1]}

    def check_category(self, state: AgentState):
        """Checks if the user-specified category exists."""
        message = state["messages"][-1]
        prompt = (
            "You are a shop assistant. Check if the category the user is asking about exists in our store. "
            "Available categories are : [Clothes, Accessories, Sneakers].Each category contains sub categories as follows: "
            "1-Clothes: ['T-shirts', 'Jeans', 'Jackets', 'Dresses'], "
            "2-Accessories: ['Watches', 'Belts', 'Hats', 'Scarves'], "
            "3-Sneakers: ['Running Shoes', 'High-tops', 'Slip-ons', 'Trainers'],if the user asked for the main category or one of the categories,respond with yes else repsond with no"
            "Respond with 'yes' or 'no' only."
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt), ("human", "Category: {category}")
        ])
        result = (prompt_template | self.LLM).invoke(message)
        return 'true' if 'yes' in result.content else 'false'

    def list_products(self, state: AgentState):
        """Lists products for the specified category."""
        category = state['messages'][-1].content
        prompt = (
            "You are a shop assistant. Based on the category provided by the user, "
            "list the products available in that category. "
            "Here are the products for each category: "
            "Clothes: ['T-shirts', 'Jeans', 'Jackets', 'Dresses'], "
            "Accessories: ['Watches', 'Belts', 'Hats', 'Scarves'], "
            "Sneakers: ['Running Shoes', 'High-tops', 'Slip-ons', 'Trainers']."
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt), ("human", "Category: {category}")
        ])
        result = (prompt_template | self.LLM).invoke(category)
        return {"messages": f"Here are the products in {category}: {result.content}\n Do you want to ask about another one?"}

    def category_not_found(self, state: AgentState):
        """Handles the case where the category is not found."""
        message =  state['messages'][-1].content
        return {"messages": "I'm sorry, we don't have that category in our store at the moment. Would you like to ask about another one?"}

    def handle_unknown_messages(self, state: AgentState):
        """Handles unknown user messages."""
        return {"messages": "I'm sorry, I am only responsible for e-commerce questions . Can you please enter the category you want?"}
    
    def run(self, config, message: str):
        """
        Runs the state graph workflow for the chatbot.
        Args:
            config: Configuration state for the graph.
            message (str): The input message from the user.
        Returns:
            str: The chatbot's response.
        """
        print(f"User messege {message}")
        current_values = self.graph.get_state(config)
        
        # Check if previous messages exist
        if 'messages' in current_values.values:
            # print("here")
            _id = current_values.values['messages'][-1].id
            updated_message = HumanMessage(content=message, id=_id)
            current_values.values['messages'][-1] = updated_message

            # Update graph state and invoke graph
            self.graph.update_state(config, current_values.values)
            result = self.graph.invoke(None, config)
            print(f"Result output {result}")
            return result['messages'][-1].content
        
        # If no messages exist, initialize with the new message
        # print("here 2")
        messages = [HumanMessage(content=message)]
        result = self.graph.invoke({'messages': messages}, config)
        return result['messages'][-1].content

    # def confirm_choice(self,state:AgentState):
    #     """Asks for user confirmation"""
    #     return {"messages":"Do you want to confirm this choice?"}
    
    # def choose_product(self,state:AgentState):
    #     """Asks the user to choose a product"""
    #     return {"messages":"Do you want to confirm this choice?"}

    
