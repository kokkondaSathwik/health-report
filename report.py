import os
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Streamlit theme
st.set_page_config(
    page_title="Health Report Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        .workflow-container {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            margin: 10px;
        }
        
        .workflow-step {
            background-color: #1E1E1E;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 3px solid #00CA51;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .user-message {
            background-color: #262730;
            border-left: 5px solid #00CA51;
        }
        
        .assistant-message {
            background-color: #1E1E1E;
            border-left: 5px solid #0078FF;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            border-radius: 4px;
            padding: 8px 16px;
            color: #FAFAFA;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #404040;
        }
        
        .stProgress > div > div {
            background-color: #00CA51;
        }
        
        .agent-card {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #dee2e6;
            background-color: black;
        }
    </style>
""", unsafe_allow_html=True)

@dataclass
class AgentResponse:
    """Structure for agent responses"""
    agent_name: str
    content: str
    confidence: float
    processing_time: float

class AgentStatus:
    """Enhanced agent status management with sidebar display"""
    def __init__(self):
        self.sidebar_placeholder = None
        self.agents = {
            'document_processor': {'status': 'idle', 'progress': 0, 'message': ''},
            'positive_analyzer': {'status': 'idle', 'progress': 0, 'message': ''},
            'negative_analyzer': {'status': 'idle', 'progress': 0, 'message': ''},
            'summary_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'recommendation_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'chat_assistant': {'status': 'idle', 'progress': 0, 'message': ''}  # Added chat assistant
        }
        
    def initialize_sidebar_placeholder(self):
        """Initialize the sidebar placeholder"""
        with st.sidebar:
            st.markdown("## 🤖 Agent Status")
            self.sidebar_placeholder = st.empty()
    
    def update_status(self, agent_name: str, status: str, progress: float, message: str = ""):
        """Update agent status and refresh sidebar display"""
        self.agents[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message
        }
        self._render_status()

    def _render_status(self):
        """Render status in sidebar"""
        if self.sidebar_placeholder is None:
            self.initialize_sidebar_placeholder()
            
        with self.sidebar_placeholder.container():
            for agent_name, status in self.agents.items():
                self._render_agent_card(agent_name, status)

    def _render_agent_card(self, agent_name: str, status: dict):
        """Render individual agent status card in sidebar"""
        colors = {
            'idle': '#6c757d',
            'working': '#007bff',
            'completed': '#28a745',
            'error': '#dc3545'
        }
        color = colors.get(status['status'], colors['idle'])
        
        # Add robot emoji to agent name
        display_name = f"{agent_name.replace('_', ' ').title()}"
        
        st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
            ">
                <div style="color: {color}; font-weight: bold;">
                    {display_name}
                </div>
                <div style="
                    color: #CCCCCC;
                    font-size: 0.8rem;
                    margin: 0.3rem 0;
                ">
                    {status['message'] or status['status'].title()}
                </div>
                <div style="
                    height: 4px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 2px;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {status['progress'] * 100}%;
                        height: 100%;
                        background-color: {color};
                        border-radius: 2px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

class HealthReportAnalyzer:
    """Enhanced health report analysis system with specialized agents"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama3-8b-8192",
            groq_api_key=GROQ_API_KEY
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize specialized medical analysis agents"""
        self.agents = {
            'document_processor': self._create_agent("""
                You are a medical document processor specialized in health reports.
                Extract all relevant medical information, organize it clearly, and maintain accuracy.
                Focus on blood work, vital signs, and other measurable health metrics.
            """),
            
            'positive_analyzer': self._create_agent("""
                You are a positive health findings specialist.
                Identify and explain all positive health indicators in the report.
                
                IMPORTANT FORMATTING INSTRUCTIONS:
                1. Each finding MUST start on a new line with a checkmark symbol (✓)
                2. After each finding value and range, add a new line with two spaces of indentation for the significance
                3. Add a blank line between each complete finding
                
                Format each finding exactly like this:
                
                ✓ [Test Name]: [Value] [Unit] (normal range: [range])
                  Significance: [Brief explanation of why this is positive]
                
                [blank line here]
                ✓ [Next Test Name]: [Value] [Unit] (normal range: [range])
                  Significance: [Brief explanation of why this is positive]
                
                Example:
                ✓ Hemoglobin: 14.5 g/dL (normal range: 13.0-17.0 g/dL)
                  Significance: Excellent oxygen-carrying capacity, indicating good red blood cell function
                
                ✓ White Blood Cells: 7.2 x 10^9/L (normal range: 4.0-10.0 x 10^9/L)
                  Significance: Strong immune system function within optimal range
                
                Ensure each finding has:
                - Checkmark symbol at start
                - Clear test name and value
                - Normal range in parentheses
                - Significance on next line with indentation
                - Blank line after each complete finding
            """),
            
            'negative_analyzer': self._create_agent("""
                You are a health risk assessment specialist.
                Identify concerning findings and potential health risks.
                Format findings as bullet points starting with "⚠".
                Each finding must be on a new line.
                Include severity levels and recommended actions.
            """),
            
            'summary_agent': self._create_agent("""
                You are a medical report summarizer.
                Create a comprehensive yet concise summary of all findings.
                Include key metrics, trends, and important observations.
                Use clear, patient-friendly language.
                Format with clear sections and bullet points.
            """),
            
            'recommendation_agent': self._create_agent("""
                You are a healthcare recommendations specialist.
                Provide actionable advice based on the report findings.
                Include lifestyle, diet, and exercise recommendations.
                Prioritize suggestions by importance and urgency.
                Format each recommendation on a new line with clear categorization.
            """)
        }

    def _create_agent(self, system_prompt: str):
        """Create an agent with specific system prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return prompt | self.llm | StrOutputParser()

    def _format_findings(self, response: str) -> str:
        """Format the findings to ensure proper line breaks and spacing"""
        # First, ensure we're working with a string
        if not isinstance(response, str):
            return str(response)

        # Split on checkmark symbol, preserving the symbol
        findings = [f.strip() for f in response.split('✓') if f.strip()]
        
        formatted_findings = []
        for finding in findings:
            # Split finding into main result and significance (if present)
            lines = [line.strip() for line in finding.split('\n') if line.strip()]
            
            if lines:
                # Format main finding line
                main_finding = lines[0]
                formatted_finding = [f"✓ {main_finding}"]
                
                # Format significance and any additional lines
                for line in lines[1:]:
                    if line.startswith('Significance:'):
                        formatted_finding.append(f"  {line}")
                    else:
                        formatted_finding.append(f"  {line}")
                
                # Join the lines for this finding
                formatted_findings.append('\n'.join(formatted_finding))
        
        # Join all findings with double newlines for spacing
        return '\n\n'.join(formatted_findings)

    async def process_document(self, file_content: str):
        """Process document and create vector store"""
        try:
            chunks = self.text_splitter.split_text(file_content)
            
            if not chunks:
                raise ValueError("No text chunks were created from the document")

            self.vectorstore = await FAISS.afrom_texts(
                texts=chunks,
                embedding=self.embeddings,
                normalize_L2=True
            )
            
            return chunks
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            raise

    async def analyze_report(self, report_text: str, agent_status: AgentStatus):
        """Analyze report using multiple agents with dynamic status updates"""
        results = {}
        
        agent_status.update_status(
            'document_processor',
            'working',
            0.0,
            'Processing document...'
        )
        
        try:
            await self.process_document(report_text)
            agent_status.update_status(
                'document_processor',
                'completed',
                1.0,
                'Document processed'
            )
            
            agents_list = list(self.agents.items())
            for idx, (agent_name, agent) in enumerate(agents_list[1:], 1):
                agent_status.update_status(
                    agent_name,
                    'working',
                    0.0,
                    f'Starting analysis...'
                )
                
                start_time = time.time()
                
                try:
                    # Get relevant context
                    if self.vectorstore is not None:
                        relevant_docs = await self.vectorstore.asimilarity_search(
                            agent_name,
                            k=3
                        )
                        context = "\n".join(doc.page_content for doc in relevant_docs)
                        augmented_text = f"Context: {context}\n\nReport: {report_text}"
                    else:
                        augmented_text = report_text
                    
                    agent_status.update_status(
                        agent_name,
                        'working',
                        0.5,
                        'Analyzing content...'
                    )
                    
                    # Get response and apply formatting for positive_analyzer
                    response = await agent.ainvoke({"input": augmented_text})
                    if agent_name == 'positive_analyzer':
                        response = self._format_findings(response)
                    
                    processing_time = time.time() - start_time
                    
                    results[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        content=response,
                        confidence=0.9,
                        processing_time=processing_time
                    )
                    
                    agent_status.update_status(
                        agent_name,
                        'completed',
                        1.0,
                        'Analysis complete'
                    )
                    
                except Exception as e:
                    results[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        content=f"Error: {str(e)}",
                        confidence=0.0,
                        processing_time=0.0
                    )
                    
                    agent_status.update_status(
                        agent_name,
                        'error',
                        1.0,
                        f'Error: {str(e)}'
                    )
            
            return results
        except Exception as e:
            st.error(f"Error in analyze_report: {str(e)}")
            raise

    async def generate_chat_response(self, query: str, context: str) -> str:
        """Generate chat response using RAG"""
        try:
            if self.vectorstore is not None:
                relevant_chunks = await self.vectorstore.asimilarity_search(
                    query,
                    k=3
                )
                additional_context = "\n".join(chunk.page_content for chunk in relevant_chunks)
                full_context = f"{context}\n\nAdditional Context: {additional_context}"
            else:
                full_context = context

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical report assistant. Use the provided context to:
                    1. Answer questions accurately
                    2. Explain medical terms simply
                    3. Provide evidence-based responses
                    4. Maintain a helpful, professional tone"""),
                ("human", "{query}"),
                ("system", "Context: {context}")
            ])
            
            chain = chat_prompt | self.llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "query": query,
                "context": full_context
            })
            
            return response
        except Exception as e:
            error_message = f"Error generating chat response: {str(e)}"
            st.error(error_message)
            return f"I apologize, but I encountered an error: {str(e)}"
def handle_chat_input():
    """Handle chat input and response"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0
    
    if "processing_message" not in st.session_state:
        st.session_state.processing_message = False
    
    # Display existing chat messages
    for message in st.session_state.chat_messages:
        if isinstance(message, HumanMessage):
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message.content}
                </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message.content}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input area
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your report:",
            key=f"chat_input_{st.session_state.chat_input_key}"
        )
    with col2:
        send_button = st.button("Send")
    
    if send_button and user_input and not st.session_state.processing_message:
        st.session_state.processing_message = True
        
        # Update chat assistant status
        st.session_state.agent_status.update_status(
            'chat_assistant',
            'working',
            0.5,
            'Processing your question...'
        )
        
        human_message = HumanMessage(content=user_input)
        st.session_state.chat_messages.append(human_message)
        
        if st.session_state.report_text:
            response = asyncio.run(
                st.session_state.analyzer.generate_chat_response(
                    user_input,
                    st.session_state.report_text
                )
            )
            
            ai_message = AIMessage(content=response)
            st.session_state.chat_messages.append(ai_message)
        
        # Update chat assistant status to completed
        st.session_state.agent_status.update_status(
            'chat_assistant',
            'completed',
            1.0,
            'Response generated'
        )
        
        st.session_state.processing_message = False
        st.session_state.chat_input_key += 1
        st.rerun()
        
def display_workflow():
    """Display the analysis workflow"""
    with st.container():
        st.markdown("""
            <div class="workflow-container">
                <h3>How It Works</h3>
                <div class="workflow-step">
                    1. Upload Your Health Report (PDF/TXT)
                </div>
                <div class="workflow-step">
                    2. AI Agents Analyze Your Report
                </div>
                <div class="workflow-step">
                    3. Get Comprehensive Analysis & Insights
                </div>
                <div class="workflow-step">
                    4. Chat with AI About Your Results
                </div>
            </div>
        """, unsafe_allow_html=True)

def main():
    """Main application with enhanced UI and dark theme"""
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = HealthReportAnalyzer()
    if 'report_results' not in st.session_state:
        st.session_state.report_results = None
    if 'report_text' not in st.session_state:
        st.session_state.report_text = None
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Health Report Analyzer")
        st.markdown("---")
        
        # File upload in sidebar
        uploaded_file = st.file_uploader(
            "Upload your health report",
            type=['pdf', 'txt'],
            key='file_uploader'
        )
        
        if uploaded_file:
            if st.button("🔍 Analyze Report", key='analyze_btn'):
                try:
                    # Read document
                    if uploaded_file.type == "application/pdf":
                        pdf_reader = PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    else:
                        text = uploaded_file.getvalue().decode()
                    
                    st.session_state.report_text = text
                    
                    # Initialize agent status display after file upload
                    st.session_state.agent_status.initialize_sidebar_placeholder()
                    
                    # Reset all agent statuses to idle
                    for agent_name in st.session_state.agent_status.agents:
                        st.session_state.agent_status.update_status(
                            agent_name,
                            'idle',
                            0.0,
                            'Waiting to start...'
                        )
                    
                    # Analyze report with dynamic status updates
                    st.session_state.report_results = asyncio.run(
                        st.session_state.analyzer.analyze_report(
                            text,
                            st.session_state.agent_status
                        )
                    )
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing report: {str(e)}")
        
        # Display agent status after file upload section
        if st.session_state.report_text:
            st.session_state.agent_status.initialize_sidebar_placeholder()
    
        # Main content area
    st.title("Health Report Analysis")
    
    if not st.session_state.report_results:
        display_workflow()
    else:
        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "✅ Positive Findings",
            "⚠️ Areas of Concern",
            "📊 Full Report",
            "💬 Chat Assistant"
        ])
        
        with tab1:
            if 'positive_analyzer' in st.session_state.report_results:
                result = st.session_state.report_results['positive_analyzer']
                st.markdown(result.content)
        
        with tab2:
            if 'negative_analyzer' in st.session_state.report_results:
                result = st.session_state.report_results['negative_analyzer']
                st.markdown(result.content)
        
        with tab3:
            if 'document_processor' in st.session_state.report_results:
                st.subheader("Document Analysis")
                st.markdown(st.session_state.report_results['document_processor'].content)
            
            if 'summary_agent' in st.session_state.report_results:
                st.subheader("Summary")
                st.markdown(st.session_state.report_results['summary_agent'].content)
            
            if 'recommendation_agent' in st.session_state.report_results:
                st.subheader("Recommendations")
                st.markdown(st.session_state.report_results['recommendation_agent'].content)
        
        with tab4:
            handle_chat_input()

if __name__ == "__main__":
    main()
