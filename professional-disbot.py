# ============================================================================
# IMPORTS
# ============================================================================
# Discord bot framework
import asyncio
import datetime
import hashlib
import io
import json
import os

# Standard library
import random
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import discord
import fitz  # PyMuPDF for image extraction
import numpy as np  # Vector operations

# External libraries
import ollama  # Local LLM integration
import requests  # Web scraping and API calls
from bs4 import BeautifulSoup  # HTML parsing
from discord import app_commands
from discord.ext import commands
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# LangChain imports for AI agents
from langchain_openai import ChatOpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer  # Semantic similarity

# ============================================================================
# CONFIGURATION
# ============================================================================
# Model configuration
myModel = "deepseek/deepseek-r1-distill-llama-70b"  # Primary text model for conversations (OpenRouter)
USE_OPENROUTER = True  # Use OpenRouter API instead of local Ollama
modelName = "Assistant"  # Display name of the bot - change this to your bot's name
imageModel = (
    "llava:7b"  # Model for image analysis (multimodal) - still uses local Ollama
)

# Bot personality configuration
BOT_SYSTEM_PROMPT = """You are {model_name}, a professional training assistant. Your role is to help users learn from official training documentation.

## Core Behavior

1. **ALWAYS cite your sources** - Every piece of information you provide MUST include the source and page number from the TRAINING DOCUMENTATION section. This is mandatory.
2. **Be direct** - No greetings, pleasantries, or filler. Just answer the question.
3. **Be accurate** - If the documentation doesn't cover something, say so. Don't guess or use outside knowledge.
4. **Continue naturally** - Use RECENT MESSAGES to maintain conversation flow. Don't repeat what you've already explained.
5. **You CAN send images** - You have access to diagrams and images from the training documentation. When users ask to "show" something, describe what you're showing and images will be sent automatically.

## Citation Format - REQUIRED

You MUST cite the source for every fact. Use this format:
- "According to the CompTIA A+ Guide (Page 92), the troubleshooting steps are..."
- "The Network+ Guide (Page 45) explains that Cat 5e supports..."

IMPORTANT: Look at the [Source: filename, Page X] tags in the TRAINING DOCUMENTATION section and cite them in your response. If you don't cite sources, your response is incomplete.

## Context Structure

Your input includes:
- **TRAINING DOCUMENTATION**: Relevant excerpts with [Source: filename, Page X] tags
- **RECENT MESSAGES**: Conversation history
- **ATTRIBUTION**: Who is asking what (in group conversations)
- **Current user**: Who you're responding to right now
- **User facts**: Background info about the user (role, experience level)

## Group Conversations

When multiple trainees are present:
- Pay attention to ATTRIBUTION to know who asked what
- Address the current user by name if needed for clarity
- Don't confuse different users' questions or statements
- Each user may be at a different point in their learning

Use documentation as the primary source. User facts help you calibrate your explanations."""

# API credentials
botToken = ""
openrouter_api_key = ""

# LangChain configuration for agents (using OpenRouter with Gemini 2.0 Flash Lite)
llm_for_agents = ChatOpenAI(
    model="google/gemini-2.0-flash-lite-001",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=1024,
)

llm_for_memory = ChatOpenAI(
    model="google/gemini-2.0-flash-lite-001",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
    max_tokens=2048,
)

# Documentation retrieval configuration
DOCS_DIRECTORY = "./training_docs"  # Directory for training documentation
VECTOR_STORE_PATH = "./vector_store"  # Persistent vector store location
IMAGES_DIRECTORY = "./training_images"  # Extracted images from PDFs

# Initialize embeddings for document retrieval (using same model as semantic similarity)
_doc_embeddings = None
_vector_store = None


def get_doc_embeddings():
    """Lazy load embeddings model for document retrieval"""
    global _doc_embeddings
    if _doc_embeddings is None:
        _doc_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        )
    return _doc_embeddings


def get_vector_store():
    """Get or create the vector store for documentation retrieval"""
    global _vector_store
    if _vector_store is None:
        embeddings = get_doc_embeddings()

        # Check if vector store already exists
        if os.path.exists(VECTOR_STORE_PATH):
            _vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings
            )
            print(f"[DOCS] Loaded existing vector store from {VECTOR_STORE_PATH}")
        else:
            # Create empty vector store - will be populated when docs are added
            _vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings
            )
            print(f"[DOCS] Created new vector store at {VECTOR_STORE_PATH}")

    return _vector_store


def load_training_documents():
    """
    Load and index training documents from the docs directory.

    Supported formats:
    - .txt (plain text)
    - .md (Markdown)
    - .pdf (PDF documents)
    - .csv (CSV files)
    """
    if not os.path.exists(DOCS_DIRECTORY):
        os.makedirs(DOCS_DIRECTORY)
        print(f"[DOCS] Created training docs directory at {DOCS_DIRECTORY}")
        return 0

    # Check if vector store already has documents
    vector_store = get_vector_store()
    try:
        existing_count = vector_store._collection.count()
        if existing_count > 0:
            print(
                f"[DOCS] Found existing vector store with {existing_count} chunks - skipping reindex"
            )
            return existing_count
    except Exception as e:
        print(f"[DOCS] Could not check existing documents: {e}")

    all_documents = []

    try:
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                DOCS_DIRECTORY, glob="**/*.txt", loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            all_documents.extend(txt_docs)
            if txt_docs:
                print(f"[DOCS] Loaded {len(txt_docs)} .txt files")
        except Exception as e:
            print(f"[DOCS] Error loading .txt files: {e}")

        # Load Markdown files
        try:
            md_loader = DirectoryLoader(
                DOCS_DIRECTORY, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
            )
            md_docs = md_loader.load()
            all_documents.extend(md_docs)
            if md_docs:
                print(f"[DOCS] Loaded {len(md_docs)} .md files")
        except Exception as e:
            print(f"[DOCS] Error loading .md files: {e}")

        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                DOCS_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            if pdf_docs:
                print(f"[DOCS] Loaded {len(pdf_docs)} .pdf files")
        except Exception as e:
            print(f"[DOCS] Error loading .pdf files: {e}")

        # Load CSV files
        try:
            csv_loader = DirectoryLoader(
                DOCS_DIRECTORY, glob="**/*.csv", loader_cls=CSVLoader
            )
            csv_docs = csv_loader.load()
            all_documents.extend(csv_docs)
            if csv_docs:
                print(f"[DOCS] Loaded {len(csv_docs)} .csv files")
        except Exception as e:
            print(f"[DOCS] Error loading .csv files: {e}")

        if not all_documents:
            print(f"[DOCS] No documents found in {DOCS_DIRECTORY}")
            return 0

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        splits = text_splitter.split_documents(all_documents)

        print(f"[DOCS] Created {len(splits)} chunks, starting embedding...")

        # Add to vector store in batches to avoid memory issues
        vector_store = get_vector_store()
        batch_size = 100  # Process 100 chunks at a time
        total_added = 0

        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            vector_store.add_documents(batch)
            total_added += len(batch)
            if total_added % 500 == 0 or total_added == len(splits):
                print(f"[DOCS] Indexed {total_added}/{len(splits)} chunks...")

        print(
            f"[DOCS] Total: {len(all_documents)} documents, {len(splits)} chunks indexed"
        )
        return len(splits)

    except Exception as e:
        print(f"[DOCS] Error loading documents: {e}")
        return 0


def retrieve_relevant_docs(query: str, k: int = 3) -> List[Dict]:
    """
    Retrieve relevant documentation chunks for a query with source metadata.

    Returns list of dicts with:
    - content: The text content
    - source: Filename/path
    - page: Page number (for PDFs)
    """
    try:
        vector_store = get_vector_store()

        # Check if vector store has any documents
        if vector_store._collection.count() == 0:
            return []

        # Perform similarity search
        docs = vector_store.similarity_search(query, k=k)

        results = []
        for doc in docs:
            result = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
            }
            # Clean up source path to just filename
            if result["source"] and "/" in result["source"]:
                result["source"] = result["source"].split("/")[-1]
            results.append(result)

        return results

    except Exception as e:
        print(f"[DOCS] Error retrieving documents: {e}")
        return []


# ============================================================================
# PDF IMAGE EXTRACTION AND RETRIEVAL
# ============================================================================

# Store image metadata for retrieval
image_metadata_store = []


def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract images from a PDF file and save them with metadata.

    Returns list of dicts with:
    - image_path: Path to saved image
    - source: PDF filename
    - page: Page number
    - context: Surrounding text for semantic search
    """
    import os

    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_DIRECTORY, exist_ok=True)

    extracted_images = []
    pdf_filename = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get page text for context
            page_text = page.get_text()

            # Get images from page
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]

                    # Extract image
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Skip very small images (likely icons/bullets)
                    if len(image_bytes) < 5000:  # Less than 5KB
                        continue

                    # Generate unique filename
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    image_filename = f"{pdf_filename}_p{page_num + 1}_img{img_index}_{image_hash}.{image_ext}"
                    image_path = os.path.join(IMAGES_DIRECTORY, image_filename)

                    # Save image
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Extract context (full page text for better semantic matching)
                    # Use the entire page text to capture what the image relates to
                    context = (
                        page_text.strip()
                        if page_text
                        else f"Image from page {page_num + 1}"
                    )

                    extracted_images.append(
                        {
                            "image_path": image_path,
                            "source": pdf_filename,
                            "page": page_num + 1,
                            "context": context,
                        }
                    )

                except Exception as img_error:
                    print(
                        f"[IMAGES] Error extracting image {img_index} from page {page_num + 1}: {img_error}"
                    )
                    continue

        doc.close()
        print(f"[IMAGES] Extracted {len(extracted_images)} images from {pdf_filename}")

    except Exception as e:
        print(f"[IMAGES] Error processing PDF {pdf_path}: {e}")

    return extracted_images


def load_pdf_images():
    """
    Load and index all images from PDF files in the training documents directory.
    Should be called during bot startup.
    """
    global image_metadata_store
    import json
    import os

    if not os.path.exists(DOCS_DIRECTORY):
        print(f"[IMAGES] Training docs directory not found: {DOCS_DIRECTORY}")
        return 0

    metadata_file = os.path.join(IMAGES_DIRECTORY, "image_metadata.json")

    # Check if we have cached metadata with full context
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                image_metadata_store = json.load(f)
            print(
                f"[IMAGES] Loaded {len(image_metadata_store)} images from metadata cache"
            )
            return len(image_metadata_store)
        except Exception as e:
            print(f"[IMAGES] Error loading metadata cache: {e}")

    # Extract images from all PDFs
    all_images = []

    for filename in os.listdir(DOCS_DIRECTORY):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(DOCS_DIRECTORY, filename)
            images = extract_images_from_pdf(pdf_path)
            all_images.extend(images)

    image_metadata_store = all_images

    # Save metadata to JSON for faster loading next time
    if all_images:
        os.makedirs(IMAGES_DIRECTORY, exist_ok=True)
        try:
            with open(metadata_file, "w") as f:
                json.dump(all_images, f)
            print(f"[IMAGES] Saved metadata cache to {metadata_file}")
        except Exception as e:
            print(f"[IMAGES] Error saving metadata cache: {e}")

    print(f"[IMAGES] Total: {len(all_images)} images extracted and indexed")

    return len(all_images)


def retrieve_relevant_images(query: str, k: int = 2) -> List[Dict]:
    """
    Retrieve images that are relevant to the query based on their page context.

    Uses semantic similarity to match query against image context.

    Returns list of dicts with:
    - image_path: Path to image file
    - source: PDF filename
    - page: Page number
    - relevance_score: How relevant the image is to the query
    """
    if not image_metadata_store:
        return []

    try:
        # Get embeddings for the query
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)

        # Calculate similarity for each image's context
        scored_images = []
        for img_meta in image_metadata_store:
            context = img_meta.get("context", "")
            if not context:
                continue

            # Get embedding for image context
            context_embedding = embeddings.embed_query(context)

            # Calculate cosine similarity
            query_vec = np.array(query_embedding)
            context_vec = np.array(context_embedding)

            similarity = np.dot(query_vec, context_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(context_vec)
            )

            scored_images.append({**img_meta, "relevance_score": float(similarity)})

        # Sort by relevance and return top k
        scored_images.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Only return images above a minimum relevance threshold
        min_relevance = 0.3
        relevant_images = [
            img for img in scored_images[:k] if img["relevance_score"] >= min_relevance
        ]

        return relevant_images

    except Exception as e:
        print(f"[IMAGES] Error retrieving relevant images: {e}")
        return []


# ============================================================================
# BOT INITIALIZATION
# ============================================================================
intents = discord.Intents.all()
intents.message_content = True
client = commands.Bot(command_prefix=".", intents=discord.Intents.all())

# ============================================================================
# BOT-TO-BOT INTERACTION TRACKING (Anti-Loop Protection)
# ============================================================================
bot_interaction_tracking = {
    "last_bot_responded_to_id": None,  # ID of last bot we responded to
    "last_bot_responded_to_timestamp": None,  # When we last responded to a bot
    "messages_since_bot_response": 0,  # Counter for cooldown
    "consecutive_bot_messages": 0,  # Track consecutive bot messages in channel
}


async def send_long_message(message, content: str):
    """
    Send a message that may exceed Discord's 2000 character limit.
    Splits into multiple messages if needed.
    """
    max_length = 2000

    if len(content) <= max_length:
        await message.reply(content)
        return

    # Split into chunks, trying to break at newlines
    chunks = []
    current_chunk = ""

    for line in content.split("\n"):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If single line is too long, split it
            if len(line) > max_length:
                while len(line) > max_length:
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                current_chunk = line + "\n"
            else:
                current_chunk = line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Send first chunk as reply, rest as follow-ups
    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)


# ============================================================================
# OPENROUTER API WRAPPER
# ============================================================================


# System prompt for the bot
def load_system_prompt():
    """Return the configured system prompt for the bot, formatted with the model name"""
    return BOT_SYSTEM_PROMPT.format(model_name=modelName)


def openrouter_chat(model: str, messages: list) -> dict:
    """
    Wrapper function to call OpenRouter API with Ollama-compatible interface.
    Returns a dict matching Ollama's response format: {'message': {'content': '...'}}
    """
    try:
        # Load system prompt and prepend it to messages
        system_prompt = load_system_prompt()

        # Build messages with system prompt first
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/yourusername/discord-bot",  # Optional but recommended
                "X-Title": "Professional Discord Bot",  # Optional but recommended
            },
            json={
                "model": model,
                "messages": full_messages,
                "temperature": 0.62,  # Low but allows some personality variation
                "top_p": 0.94,  # High for accuracy
                "repetition_penalty": 1.16,  # Moderate
                "frequency_penalty": 0.03,  # Low
                "presence_penalty": 0.06,  # Low to stay focused
                "provider": {
                    "order": [
                        "Together"
                    ],  # Use Together provider specifically (no logging/tracking)
                    "allow_fallbacks": False,  # Don't fall back to other providers
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        # Convert OpenRouter format to Ollama format
        return {"message": {"content": data["choices"][0]["message"]["content"]}}
    except requests.exceptions.RequestException as e:
        print(f"[OPENROUTER ERROR] Failed to get response: {e}")
        # Fallback to a simple error message
        return {
            "message": {
                "content": "I apologize, but I'm experiencing technical difficulties. Please try again."
            }
        }


# ============================================================================
# CHAT LOGS - SHORT-TERM MEMORY (Immediate Conversation Context)
# ============================================================================
# NOTE: For group channels, we use channel_history as the SINGLE SOURCE OF TRUTH
# and build LLM chatlogs dynamically from it. This prevents dual-system confusion.
# For DMs, we still use chatlogDM since channel_history is for group channels only.
chatlogDM = []  # Direct message conversation history (DMs only)
filterlog = []  # Filter log for processing search/image results

# Configuration for chatlog size management
MAX_CHATLOG_MESSAGES = (
    100  # Maximum messages to keep in chatlog (increased for 70B model)
)

# ============================================================================
# USER MEMORY SYSTEM - LONG-TERM MEMORY (Persistent Facts & History)
# ============================================================================

# Initialize semantic similarity model (lazy load on first use)
_semantic_model = None


def get_semantic_model():
    """Lazy load the sentence transformer model"""
    global _semantic_model
    if _semantic_model is None:
        # print("[SEMANTIC] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        # print("[SEMANTIC] Model loaded successfully")
    return _semantic_model


class UserMemorySystem:
    """
    Manages LONG-TERM persistent user memories including facts, preferences, and task history.
    This is separate from chatlog (SHORT-TERM/immediate context).

    SHORT-TERM: Recent conversation in chatlog (last 40 messages)
    LONG-TERM: Persistent facts, past events, past tasks stored here

    Supports username aliases (e.g., "sylvester" and "sylv" are the same person)
    Uses semantic similarity for intelligent fact retrieval
    """

    def __init__(self):
        self.memory_file = "disbot_user_memories.json"  # Persistent storage file
        self.memories = self.load_memories()  # Load from disk on startup

    def categorize_fact(self, fact_text: str) -> str:
        """
        Auto-categorize a fact based on keyword patterns.

        Categories:
        - professional: work, job, career related
        - personality: character traits, behaviors, tendencies
        - preferences: likes, dislikes, interests
        - technical: skills, tools, technologies
        - history: past events, background information
        - general: uncategorized/other
        """
        text_lower = fact_text.lower()

        # Professional indicators (work, career)
        if any(
            word in text_lower
            for word in [
                "work",
                "job",
                "career",
                "company",
                "team",
                "project",
                "manager",
                "developer",
                "engineer",
                "designer",
                "analyst",
                "consultant",
                "role",
                "position",
                "department",
                "office",
            ]
        ):
            return "professional"

        # Preference indicators (what they like/dislike)
        if any(
            word in text_lower
            for word in [
                "likes",
                "dislikes",
                "prefers",
                "favorite",
                "enjoys",
                "hates",
                "loves to",
                "interests include",
                "can't stand",
                "passionate about",
            ]
        ):
            return "preferences"

        # Technical indicators (skills, tools)
        if any(
            word in text_lower
            for word in [
                "python",
                "javascript",
                "code",
                "programming",
                "software",
                "database",
                "api",
                "framework",
                "library",
                "tool",
                "technology",
                "system",
                "platform",
                "language",
                "skill",
            ]
        ):
            return "technical"

        # History indicators (past events, background)
        if any(
            word in text_lower
            for word in [
                "met ",
                "was ",
                "used to",
                "ago",
                "in 19",
                "in 20",
                "grew up",
                "born",
                "childhood",
                "graduated",
                "worked at",
                "lived in",
                "previously",
                "former",
                "past",
                "history",
            ]
        ):
            return "history"

        # Personality indicators (traits, behaviors)
        if any(
            word in text_lower
            for word in [
                "is ",
                "tends to",
                "usually",
                "often",
                "always",
                "never",
                "personality",
                "character",
                "trait",
                "behaves",
                "acts",
                "outgoing",
                "shy",
                "confident",
                "friendly",
                "professional",
            ]
        ):
            return "personality"

        # Default to general if no clear match
        return "general"

    def resolve_username(self, username: str) -> str:
        """
        Resolve a username to its canonical form using aliases.
        If the username has aliases defined, return the canonical name.
        Otherwise, return the normalized username.
        """
        username_key = username.lower()

        # Check if this username exists directly
        if username_key in self.memories:
            return username_key

        # Check if this username is an alias for another user
        for canonical_name, data in self.memories.items():
            if "aliases" in data and username_key in [
                alias.lower() for alias in data["aliases"]
            ]:
                return canonical_name

        # Not found, return the normalized form
        return username_key

    def load_memories(self):
        """Load user memories from JSON file"""
        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("Creating new memory file...")
            return {}

    def save_memories(self):
        """Save memories to JSON file"""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memories, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save memories: {e}")
            return False

    def get_user_facts(self, username: str) -> List[str]:
        """Get all stored facts about a specific user (returns text only for compatibility)"""
        username_key = self.resolve_username(username)
        if username_key in self.memories:
            facts = self.memories[username_key].get("facts", [])
            # Handle both old (string) and new (dict) formats
            return [f if isinstance(f, str) else f.get("text", "") for f in facts]
        return []

    def get_user_facts_with_timestamps(self, username: str) -> List[Dict]:
        """Get all stored facts with their timestamps and categories"""
        username_key = self.resolve_username(username)
        if username_key in self.memories:
            facts = self.memories[username_key].get("facts", [])
            # Migrate old formats to new dict format with category
            result = []
            for f in facts:
                if isinstance(f, str):
                    # Old string format - categorize it
                    result.append(
                        {
                            "text": f,
                            "timestamp": "unknown",
                            "category": self.categorize_fact(f),
                        }
                    )
                elif "category" not in f:
                    # Dict format but missing category
                    result.append(
                        {
                            "text": f.get("text", ""),
                            "timestamp": f.get("timestamp", "unknown"),
                            "category": self.categorize_fact(f.get("text", "")),
                        }
                    )
                else:
                    # Already has all fields
                    result.append(f)
            return result
        return []

    def add_user_fact(self, username: str, fact: str) -> bool:
        """Add a new fact about a user with semantic deduplication"""
        username_key = self.resolve_username(username)

        # Initialize user memory if needed
        if username_key not in self.memories:
            self.memories[username_key] = {"facts": [], "tasks": []}

        existing_facts = self.memories[username_key].get("facts", [])

        # Extract text from existing facts (handle both string and dict formats)
        existing_fact_texts = []
        for f in existing_facts:
            if isinstance(f, str):
                existing_fact_texts.append(f)
            else:
                existing_fact_texts.append(f.get("text", ""))

        # Exact duplicate check
        if fact in existing_fact_texts:
            print(f"[MEMORY] Skipping exact duplicate: '{fact[:50]}...'")
            return False

        # Semantic similarity check to avoid near-duplicates
        try:
            if len(existing_fact_texts) > 0:
                model = get_semantic_model()

                # Encode new fact and existing facts
                new_fact_embedding = model.encode([fact])[0]
                existing_embeddings = model.encode(existing_fact_texts)

                # Check similarity with each existing fact
                for i, existing_embedding in enumerate(existing_embeddings):
                    similarity = np.dot(new_fact_embedding, existing_embedding) / (
                        np.linalg.norm(new_fact_embedding)
                        * np.linalg.norm(existing_embedding)
                    )

                    # If very similar (>0.85), it's essentially a duplicate
                    if similarity > 0.85:
                        print(
                            f"[MEMORY] Skipping semantic duplicate (similarity: {similarity:.3f})"
                        )
                        print(f"[MEMORY] New: '{fact[:50]}...'")
                        print(f"[MEMORY] Existing: '{existing_fact_texts[i][:50]}...'")
                        # Replace the old one with the new one if it's more detailed
                        if len(fact) > len(existing_fact_texts[i]):
                            print(f"[MEMORY] Replacing with more detailed version")
                            timestamp = datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M"
                            )
                            category = self.categorize_fact(fact)
                            self.memories[username_key]["facts"][i] = {
                                "text": fact,
                                "timestamp": timestamp,
                                "category": category,
                            }
                            print(f"[MEMORY] Categorized as: {category}")
                            return self.save_memories()
                        return False

                    # If moderately similar (0.7-0.85), consider merging
                    elif similarity > 0.7:
                        print(
                            f"[MEMORY] Similar fact detected (similarity: {similarity:.3f})"
                        )
                        print(f"[MEMORY] New: '{fact[:50]}...'")
                        print(f"[MEMORY] Existing: '{existing_fact_texts[i][:50]}...'")
                        # Keep both but note the similarity

        except Exception as e:
            print(f"[MEMORY] Semantic deduplication failed, using exact matching: {e}")

        # Add the new fact with timestamp and category
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        category = self.categorize_fact(fact)
        self.memories[username_key]["facts"].append(
            {"text": fact, "timestamp": timestamp, "category": category}
        )
        print(f"[MEMORY] Added new fact: '{fact[:60]}...'")
        print(f"[MEMORY] Categorized as: {category}")

        return self.save_memories()

    def _prune_facts_intelligently(self, facts: list, max_facts: int = 15) -> list:
        """
        Prune facts using semantic similarity to keep the most diverse and informative ones.
        Instead of just keeping the most recent, keep a diverse set that covers different topics.
        Handles both old (string) and new (dict) fact formats.
        """
        if len(facts) <= max_facts:
            return facts

        try:
            model = get_semantic_model()

            # Extract text from facts (handle both string and dict formats)
            fact_texts = []
            for f in facts:
                if isinstance(f, str):
                    fact_texts.append(f)
                else:
                    fact_texts.append(f.get("text", ""))

            # Encode all fact texts
            embeddings = model.encode(fact_texts)

            # Keep facts that are semantically diverse
            kept_facts = []
            kept_embeddings = []

            # Always keep the most recent fact
            kept_facts.append(facts[-1])
            kept_embeddings.append(embeddings[-1])

            # Iterate through facts from newest to oldest (excluding the last one we just added)
            for i in range(len(facts) - 2, -1, -1):
                if len(kept_facts) >= max_facts:
                    break

                fact_embedding = embeddings[i]

                # Check if this fact is sufficiently different from already kept facts
                is_diverse = True
                for kept_embedding in kept_embeddings:
                    similarity = np.dot(fact_embedding, kept_embedding) / (
                        np.linalg.norm(fact_embedding) * np.linalg.norm(kept_embedding)
                    )

                    # If too similar to an existing kept fact, skip it
                    if similarity > 0.85:
                        is_diverse = False
                        print(
                            f"[MEMORY PRUNING] Removing similar fact: '{fact_texts[i][:50]}...'"
                        )
                        break

                if is_diverse:
                    kept_facts.insert(
                        0, facts[i]
                    )  # Insert at beginning to maintain order
                    kept_embeddings.insert(0, fact_embedding)

            print(f"[MEMORY PRUNING] Kept {len(kept_facts)}/{len(facts)} diverse facts")
            return kept_facts

        except Exception as e:
            print(f"[MEMORY PRUNING] Semantic pruning failed, keeping most recent: {e}")
            # Fallback: just keep the most recent facts
            return facts[-max_facts:]

    def get_relevant_facts(
        self, username: str, context: str, max_facts: int = 3
    ) -> List[str]:
        """
        Get facts about a user that are relevant to the current context.
        Uses semantic similarity with category-aware boosting for better context relevance.
        """
        user_facts_with_meta = self.get_user_facts_with_timestamps(username)
        if not user_facts_with_meta:
            return []

        # Detect what type of question is being asked and boost relevant categories
        context_lower = context.lower()
        category_boosts = {}

        # Preference-related queries
        if any(
            word in context_lower
            for word in ["like", "favorite", "prefer", "enjoy", "interest", "hobby"]
        ):
            category_boosts["preferences"] = 0.15

        # Personality/character queries
        if any(
            word in context_lower
            for word in [
                "who",
                "personality",
                "character",
                "type of person",
                "kind of",
                "describe",
            ]
        ):
            category_boosts["personality"] = 0.15

        # Technical/skills queries
        if any(
            word in context_lower
            for word in [
                "skill",
                "technology",
                "programming",
                "tool",
                "code",
                "technical",
            ]
        ):
            category_boosts["technical"] = 0.15

        # Professional/work queries
        if any(
            word in context_lower
            for word in ["work", "job", "career", "project", "team", "company"]
        ):
            category_boosts["professional"] = 0.15

        # History/background queries
        if any(
            word in context_lower
            for word in [
                "when",
                "where",
                "met",
                "first",
                "history",
                "past",
                "before",
                "remember",
            ]
        ):
            category_boosts["history"] = 0.15

        # Use semantic similarity to find relevant facts
        try:
            model = get_semantic_model()

            # Extract fact texts for encoding
            fact_texts = [f["text"] for f in user_facts_with_meta]

            # Encode the context and all facts
            context_embedding = model.encode([context])[0]
            fact_embeddings = model.encode(fact_texts)

            # Calculate cosine similarity with category boosting
            scored_facts = []
            for i, fact_embedding in enumerate(fact_embeddings):
                # Base semantic similarity
                similarity = np.dot(context_embedding, fact_embedding) / (
                    np.linalg.norm(context_embedding) * np.linalg.norm(fact_embedding)
                )

                # Apply category boost if applicable
                category = user_facts_with_meta[i].get("category", "general")
                category_boost = category_boosts.get(category, 0)
                final_score = similarity + category_boost

                scored_facts.append((user_facts_with_meta[i], final_score, similarity))

            # Sort by final score (higher is more relevant)
            scored_facts.sort(key=lambda x: x[1], reverse=True)

            # Filter out facts with low base similarity (< 0.3 threshold)
            # and facts that are too similar to the input message (> 0.85 threshold to avoid circular context)
            relevant_facts = []
            for fact_meta, final_score, base_similarity in scored_facts:
                if (
                    0.3 < base_similarity < 0.85
                ):  # Sweet spot: relevant but not redundant
                    relevant_facts.append(fact_meta["text"])
                    # Uncomment for debugging:
                    # category = fact_meta.get('category', 'general')
                    # print(f"[SEMANTIC] [{category}] {final_score:.3f} (base: {base_similarity:.3f}) - '{fact_meta['text'][:50]}...'")

            # Return top N facts (as text strings for compatibility)
            return relevant_facts[:max_facts]

        except Exception as e:
            print(
                f"[SEMANTIC] Error using semantic similarity, falling back to keyword matching: {e}"
            )
            # Fallback to simple keyword matching if semantic model fails
            relevant_facts = []
            context_lower = context.lower()
            fact_texts = [f["text"] for f in user_facts_with_meta]
            for fact_text in fact_texts[-5:]:  # Only check recent facts
                if any(
                    word in fact_text.lower()
                    for word in context_lower.split()
                    if len(word) > 4
                ):
                    relevant_facts.append(fact_text)
            return relevant_facts[:max_facts]

    def add_task_memory(
        self,
        username: str,
        task_type: str,
        request: str,
        outcome: str,
        timestamp: str = None,
    ) -> bool:
        """Add a task memory (what user asked for and what happened)"""
        username_key = self.resolve_username(username)

        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        task_memory = {
            "type": task_type,  # 'internet', 'standard'
            "request": request,
            "outcome": outcome,
            "timestamp": timestamp,
        }

        if username_key not in self.memories:
            self.memories[username_key] = {"facts": [], "tasks": []}
        elif "tasks" not in self.memories[username_key]:
            self.memories[username_key]["tasks"] = []

        self.memories[username_key]["tasks"].append(task_memory)

        return self.save_memories()

    def get_user_tasks(self, username: str) -> List[Dict]:
        """Get all stored tasks for a specific user"""
        username_key = self.resolve_username(username)
        if username_key in self.memories:
            return self.memories[username_key].get("tasks", [])
        return []

    def get_relevant_tasks(
        self, username: str, context: str, max_tasks: int = 3
    ) -> List[Dict]:
        """Get tasks relevant to the current context"""
        user_tasks = self.get_user_tasks(username)
        if not user_tasks:
            return []

        context_words = context.lower().split()
        relevant_tasks = []

        for task in user_tasks:
            # Calculate relevance score
            score = 0
            task_text = f"{task.get('request', '')} {task.get('outcome', '')}".lower()

            # Boost score for matching words
            score += sum(
                1 for word in context_words if word in task_text and len(word) > 2
            )

            # Boost for recent tasks
            try:
                task_time = datetime.datetime.strptime(
                    task.get("timestamp", ""), "%Y-%m-%d %H:%M"
                )
                hours_ago = (datetime.datetime.now() - task_time).total_seconds() / 3600
                if hours_ago < 24:  # Tasks from last 24 hours get boost
                    score += 2
                elif hours_ago < 168:  # Tasks from last week get smaller boost
                    score += 1
            except:
                pass

            if score > 0:
                relevant_tasks.append((task, score))

        # Sort by relevance
        relevant_tasks.sort(key=lambda x: x[1], reverse=True)
        return [task[0] for task in relevant_tasks[:max_tasks]]

    def clear_user_memories(self, username: str) -> bool:
        """Clear all memories for a specific user"""
        username_key = self.resolve_username(username)
        if username_key in self.memories:
            del self.memories[username_key]
            return self.save_memories()
        return False

    def add_event_memory(
        self,
        username: str,
        event_description: str,
        participants: List[str] = None,
        timestamp: str = None,
    ) -> bool:
        """
        Add an event memory - specific instances that occurred.
        These are different from facts (general truths) and should track specific occurrences.

        Args:
            username: The primary user this event is associated with
            event_description: Description of what happened
            participants: List of all participants in the event (e.g., ["Assistant", "user1"])
            timestamp: When the event occurred

        Examples:
        - "discussed Python project on Oct 15th, 2025"
        - "debugged code together on Oct 16th, 2025"
        - "set up development environment on Oct 10th, 2025"
        """
        username_key = self.resolve_username(username)

        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Extract participants from description if not provided
        if participants is None:
            participants = []
            # Always include the primary user
            participants.append(username_key)
            # Look for the bot name in the description
            if modelName.lower() in event_description.lower():
                participants.append(modelName)
        else:
            # Normalize participant names
            participants = [
                self.resolve_username(p)
                if p.lower() != modelName.lower()
                else modelName
                for p in participants
            ]

        event_entry = {
            "description": event_description,
            "timestamp": timestamp,
            "participants": participants,  # Track who was involved
        }

        # Initialize user memory if needed
        if username_key not in self.memories:
            self.memories[username_key] = {"facts": [], "tasks": [], "events": []}
        elif "events" not in self.memories[username_key]:
            self.memories[username_key]["events"] = []

        # Check for exact duplicates
        existing_events = self.memories[username_key].get("events", [])
        for existing in existing_events:
            if (
                existing["description"] == event_description
                and existing["timestamp"] == timestamp
            ):
                print(f"[EVENT MEMORY] Skipping exact duplicate event")
                return False

        # Add the event
        self.memories[username_key]["events"].append(event_entry)
        print(
            f"[EVENT MEMORY] Added event: '{event_description}' at {timestamp} with participants: {participants}"
        )

        return self.save_memories()

    def get_user_events(self, username: str, max_events: int = 5) -> List[Dict]:
        """Get recent event memories for a user"""
        username_key = self.resolve_username(username)
        if username_key in self.memories:
            events = self.memories[username_key].get("events", [])
            # Return most recent events first
            return events[-max_events:][::-1] if events else []
        return []

    def get_relevant_events(
        self, username: str, context: str, max_events: int = 3
    ) -> List[Dict]:
        """
        Get event memories relevant to the current context using semantic similarity.

        CRITICAL: This searches across ALL users to find events involving participants
        mentioned in the context. This will find events from user memories where the bot participated.
        """
        # Extract potential participant names from context
        context_lower = context.lower()

        # Look for known usernames in the context
        potential_participants = []
        for user_key in self.memories.keys():
            if user_key.lower() in context_lower:
                potential_participants.append(user_key)
            # Also check aliases
            aliases = self.memories[user_key].get("aliases", [])
            for alias in aliases:
                if alias.lower() in context_lower:
                    potential_participants.append(user_key)
                    break

        # Also check for bot name in context
        if modelName.lower() in context_lower:
            potential_participants.append(modelName)

        # Collect all candidate events
        candidate_events = []

        # Search across ALL users for events involving mentioned participants
        if potential_participants:
            for user_key in self.memories.keys():
                user_events = self.memories[user_key].get("events", [])
                for event in user_events:
                    event_participants = event.get("participants", [])
                    # Check if any of the potential participants are in this event
                    if any(
                        p in event_participants
                        or p.lower() in [ep.lower() for ep in event_participants]
                        for p in potential_participants
                    ):
                        candidate_events.append(
                            (event, True)
                        )  # Mark as participant match

        # Also search the current user's events
        user_events_all = self.memories.get(self.resolve_username(username), {}).get(
            "events", []
        )
        for event in user_events_all:
            candidate_events.append(
                (event, False)
            )  # Not necessarily a participant match

        # Remove duplicates
        seen_events = set()
        unique_candidates = []
        for event, is_participant_match in candidate_events:
            event_key = (event["description"], event["timestamp"])
            if event_key not in seen_events:
                seen_events.add(event_key)
                unique_candidates.append((event, is_participant_match))

        if not unique_candidates:
            return []

        # Use semantic similarity to score events
        try:
            model = get_semantic_model()

            # Encode context
            context_embedding = model.encode([context])[0]

            # Extract event descriptions and encode them
            event_descriptions = [
                event["description"] for event, _ in unique_candidates
            ]
            event_embeddings = model.encode(event_descriptions)

            # Calculate semantic similarity scores
            scored_events = []
            for i, (event, is_participant_match) in enumerate(unique_candidates):
                # Calculate cosine similarity
                similarity = np.dot(context_embedding, event_embeddings[i]) / (
                    np.linalg.norm(context_embedding)
                    * np.linalg.norm(event_embeddings[i])
                )

                # Boost score if participant was mentioned
                if is_participant_match:
                    similarity += 0.2  # Boost by 0.2 for participant matches

                scored_events.append((event, similarity))

            # Sort by similarity score
            scored_events.sort(key=lambda x: x[1], reverse=True)

            # Filter for relevance (similarity > 0.2) and return top N
            relevant_events = [
                (event, score) for event, score in scored_events if score > 0.2
            ]
            result = [event for event, score in relevant_events[:max_events]]

            # if result:
            #     print(f"[EVENT SEARCH] Found {len(result)} semantically relevant events")
            #     for event in result:
            #         print(f"[EVENT SEARCH]   - {event['description']} (participants: {event.get('participants', [])})")

            return result

        except Exception as e:
            print(
                f"[SEMANTIC] Error using semantic similarity for events, falling back to keyword matching: {e}"
            )
            # Fallback to keyword matching
            context_words = set(word for word in context_lower.split() if len(word) > 3)
            scored_events = []
            for event, is_participant_match in unique_candidates:
                event_desc = event["description"].lower()
                matches = sum(1 for word in context_words if word in event_desc)
                score = matches + (5 if is_participant_match else 0)
                if score > 0:
                    scored_events.append((event, score))

            scored_events.sort(key=lambda x: x[1], reverse=True)
            return [event for event, score in scored_events[:max_events]]

    def deduplicate_all_facts(self) -> Dict[str, int]:
        """
        Deduplicate facts for all users using semantic similarity.
        Returns a dict of username -> number of facts removed.
        """
        results = {}

        for username in self.memories.keys():
            facts = self.memories[username].get("facts", [])
            if len(facts) <= 1:
                continue

            original_count = len(facts)

            # Use semantic pruning to remove duplicates
            deduplicated = self._prune_facts_intelligently(facts, max_facts=15)
            self.memories[username]["facts"] = deduplicated

            removed = original_count - len(deduplicated)
            if removed > 0:
                results[username] = removed
                print(
                    f"[MEMORY CLEANUP] {username}: removed {removed} duplicate/similar facts ({original_count} -> {len(deduplicated)})"
                )

        if results:
            self.save_memories()

        return results


# Initialize memory system
user_memory = UserMemorySystem()


# ============================================================================
# CHANNEL HISTORY TRACKING
# ============================================================================
class ChannelHistory:
    """
    Tracks recent message history in group channels for context awareness.
    Messages are stored in-memory and summarized when the bot is mentioned.
    """

    def __init__(self, max_messages=50):
        # Dictionary structure: channel_id -> {'messages': [], 'last_summary': None, 'last_fact_extraction': 0}
        self.channels = {}
        self.max_messages = max_messages  # Maximum messages to store per channel

    def add_message(
        self,
        channel_id: int,
        username: str,
        content: str,
        timestamp: str = None,
        metadata: dict = None,
    ):
        """Add a message to channel history with optional rich metadata"""
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        if channel_id not in self.channels:
            self.channels[channel_id] = {
                "messages": [],
                "last_summary": None,
                "last_fact_extraction": 0,
            }

        # Use metadata if provided (from enriched messages), otherwise create basic message
        if metadata:
            message_data = metadata
        else:
            message_data = {
                "username": username,
                "content": content,
                "timestamp": timestamp,
            }

        self.channels[channel_id]["messages"].append(message_data)

        # Keep only last N messages to prevent memory bloat
        if len(self.channels[channel_id]["messages"]) > self.max_messages:
            self.channels[channel_id]["messages"] = self.channels[channel_id][
                "messages"
            ][-self.max_messages :]

    def get_recent_messages(self, channel_id: int, limit: int = None) -> List[Dict]:
        """Get recent messages from a channel"""
        if channel_id not in self.channels:
            return []

        messages = self.channels[channel_id]["messages"]
        if limit:
            return messages[-limit:]
        return messages

    def clear_messages(self, channel_id: int):
        """Clear message history for a channel (after summarization)"""
        if channel_id in self.channels:
            self.channels[channel_id]["messages"] = []

    def set_summary(self, channel_id: int, summary: str):
        """Store the last summary for a channel"""
        if channel_id not in self.channels:
            self.channels[channel_id] = {
                "messages": [],
                "last_summary": None,
                "last_fact_extraction": 0,
            }
        self.channels[channel_id]["last_summary"] = summary

    def should_extract_facts(
        self, channel_id: int, message_threshold: int = 20
    ) -> bool:
        """Check if enough messages have passed since last fact extraction"""
        if channel_id not in self.channels:
            return True

        current_message_count = len(self.channels[channel_id]["messages"])
        last_extraction_count = self.channels[channel_id].get("last_fact_extraction", 0)

        # Extract facts every N messages
        return (current_message_count - last_extraction_count) >= message_threshold

    def mark_facts_extracted(self, channel_id: int):
        """Mark that facts have been extracted at this point"""
        if channel_id in self.channels:
            self.channels[channel_id]["last_fact_extraction"] = len(
                self.channels[channel_id]["messages"]
            )

    def get_summary(self, channel_id: int) -> str:
        """Get the last summary for a channel"""
        if channel_id in self.channels:
            return self.channels[channel_id].get("last_summary")
        return None


# Initialize channel history tracker
channel_history = ChannelHistory(
    max_messages=150
)  # Increased for 70B model's better context handling


# ============================================================================
# CHATLOG MANAGEMENT FUNCTIONS
# ============================================================================
def build_chatlog_from_channel_history(channel_id: int, limit: int = 16) -> list:
    """
    Build LLM-compatible chatlog from channel_history (single source of truth).

    This dynamically converts enriched channel_history messages into the format
    expected by the LLM: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

    IMPORTANT: Only messages from THIS specific bot are marked as 'assistant'.
    Messages from OTHER bots (like Storm) are treated as 'user' messages with attribution.

    Args:
        channel_id: The Discord channel ID
        limit: Maximum number of messages to include (default: 40)

    Returns:
        List of message dicts in LLM format
    """
    recent_messages = channel_history.get_recent_messages(channel_id, limit=limit)

    chatlog = []
    for msg_data in recent_messages:
        username = msg_data.get("username", "Unknown")
        content = msg_data.get("content", "")
        author_id = msg_data.get("author_id")

        # Determine role based on whether it's THIS specific bot
        # Check author_id against client.user.id to distinguish THIS bot from other bots
        if client.user and author_id == client.user.id:
            # This message is from THIS bot - mark as assistant
            chatlog.append({"role": "assistant", "content": content})
        else:
            # This message is from a user OR another bot - mark as user with attribution
            chatlog.append({"role": "user", "content": f"{username}: {content}"})

    return chatlog


async def enrich_message_context(msg) -> dict:
    """
    Extract rich contextual metadata from a Discord message to help understand
    WHO is talking TO whom and what they're discussing.
    """
    # Clean content by replacing mentions with display names
    clean_content = msg.content

    # Replace user mentions with display names
    for mentioned_user in msg.mentions:
        clean_content = clean_content.replace(
            f"<@{mentioned_user.id}>", mentioned_user.display_name
        )
        clean_content = clean_content.replace(
            f"<@!{mentioned_user.id}>", mentioned_user.display_name
        )

    # Replace role mentions with role names
    for mentioned_role in msg.role_mentions:
        clean_content = clean_content.replace(
            f"<@&{mentioned_role.id}>", f"@{mentioned_role.name}"
        )

    # Replace channel mentions with channel names
    if hasattr(msg, "channel_mentions"):
        for mentioned_channel in msg.channel_mentions:
            clean_content = clean_content.replace(
                f"<#{mentioned_channel.id}>", f"#{mentioned_channel.name}"
            )

    # Build enriched message data
    msg_data = {
        "username": msg.author.display_name,
        "content": clean_content,
        "timestamp": msg.created_at.strftime("%Y-%m-%d %H:%M"),
        "is_bot": msg.author.bot,
        "author_id": msg.author.id,  # Store author ID to distinguish between different bots
        "reply_to": None,
        "mentioned_users": [],
        "has_attachment": False,
        "attachment_types": [],
        "reactions": [],
    }

    # Track if this is a REPLY (shows WHO is responding TO whom)
    if msg.reference and msg.reference.resolved:
        replied_msg = msg.reference.resolved
        msg_data["reply_to"] = {
            "username": replied_msg.author.display_name,
            "content": replied_msg.content[:100],  # First 100 chars for context
        }

    # Track explicit mentions (shows WHO is addressing whom)
    if msg.mentions:
        msg_data["mentioned_users"] = [user.display_name for user in msg.mentions]

    # Track attachments (images, files)
    if msg.attachments:
        msg_data["has_attachment"] = True
        msg_data["attachment_types"] = [
            att.content_type or "file" for att in msg.attachments
        ]

    # Track reactions (shows engagement and sentiment)
    if msg.reactions:
        for reaction in msg.reactions:
            msg_data["reactions"].append(
                {"emoji": str(reaction.emoji), "count": reaction.count}
            )

    # Track user activities (Spotify, gaming, streaming, etc.)
    # NOTE: We exclude CustomActivity (custom status messages) as they're not real activities
    msg_data["activities"] = []
    if msg.author.activities:
        for activity in msg.author.activities:
            if isinstance(activity, discord.Spotify):
                msg_data["activities"].append(
                    {
                        "type": "spotify",
                        "song": activity.title,
                        "artist": activity.artist,
                        "album": activity.album,
                    }
                )
            elif isinstance(activity, discord.Game):
                msg_data["activities"].append({"type": "game", "name": activity.name})
            elif isinstance(activity, discord.Streaming):
                msg_data["activities"].append(
                    {
                        "type": "streaming",
                        "name": activity.name,
                        "url": activity.url if hasattr(activity, "url") else None,
                    }
                )
            # Skip CustomActivity - those are status messages, not actual activities

    return msg_data


async def fetch_recent_channel_messages(channel, limit=50, force_refresh=False):
    """
    Fetch recent messages from a Discord channel and populate channel_history.
    This is used when the bot is mentioned but has no prior context.

    Args:
        channel: Discord channel to fetch from
        limit: Number of messages to fetch (default 20)
        force_refresh: If True, always fetch fresh messages from Discord even if we have history
    """
    try:
        # Check if we already have enough messages tracked
        existing_messages = channel_history.get_recent_messages(channel.id)
        if len(existing_messages) >= 30 and not force_refresh:
            # Have enough context, don't need to fetch
            print(
                f"[DEBUG] Using {len(existing_messages)} cached messages from channel_history"
            )
            return existing_messages

        # Fetch recent messages from Discord with rich context
        # This happens if: no history, less than 30 messages, or force_refresh=True
        print(
            f"[DEBUG] Fetching from Discord (current history: {len(existing_messages)} messages)"
        )
        messages = []
        async for msg in channel.history(limit=limit):
            # Get enriched message data (include bot's own messages for context)
            msg_data = await enrich_message_context(msg)
            messages.append(msg_data)

        # Reverse to get chronological order (history() returns newest first)
        messages.reverse()

        # Clear existing history and add fresh messages
        if channel.id in channel_history.channels:
            channel_history.channels[channel.id]["messages"] = []

        # Add to channel_history (store enriched data)
        for msg_data in messages:
            channel_history.add_message(
                channel_id=channel.id,
                username=msg_data["username"],
                content=msg_data["content"],
                timestamp=msg_data["timestamp"],
                metadata=msg_data,  # Store full enriched data
            )

        print(f"[DEBUG] Fetched {len(messages)} historical messages from channel")
        return channel_history.get_recent_messages(channel.id)

    except Exception as e:
        print(f"Error fetching channel history: {e}")
        return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def export_to_json(object, filename):
    """Export a Python object to JSON file"""
    with open(filename, "w") as f:
        json.dump(object, f)


def import_from_json(filename):
    """Import a Python object from JSON file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("file not found")
        return None


def analyze_conversation_state(
    recent_messages: List[Dict], bot_name: str = "Assistant"
) -> str:
    """
    Analyze recent messages to extract conversation state:
    - Unanswered questions directed at the bot
    - Pending commitments or actions
    - Current topic flow
    Returns a brief state summary for context
    """
    if not recent_messages or len(recent_messages) < 2:
        return None

    state_notes = []

    # Get last 5 messages for state analysis
    recent_window = (
        recent_messages[-5:] if len(recent_messages) > 5 else recent_messages
    )

    # Track unanswered questions
    unanswered_questions = []
    bot_responded_after = False

    for i, msg in enumerate(recent_window):
        content = msg.get("content", "").strip()
        username = msg.get("username", "Unknown")
        is_bot = username == bot_name or msg.get("is_bot", False)

        # Check if this message contains a question mark (simple heuristic)
        if "?" in content and not is_bot:
            # Check if bot responded after this question
            bot_responded_after = False
            for later_msg in recent_window[i + 1 :]:
                if later_msg.get("username") == bot_name or later_msg.get("is_bot"):
                    bot_responded_after = True
                    break

            if not bot_responded_after:
                # Extract the question (last sentence with ?)
                sentences = content.split("?")
                if sentences and len(sentences[0].strip()) > 0:
                    question_text = sentences[0].strip().split(".")[-1].strip()
                    unanswered_questions.append(f"{username} asked: '{question_text}?'")

    # Add unanswered questions to state
    if unanswered_questions:
        state_notes.append(
            "UNANSWERED QUESTION: " + " | ".join(unanswered_questions[-2:])
        )  # Last 2 questions

    # Detect if conversation is repeating/stuck
    bot_messages = [
        m.get("content", "").lower()
        for m in recent_window
        if m.get("username") == bot_name or m.get("is_bot")
    ]
    if len(bot_messages) >= 3:
        # Check for similar patterns in bot's recent responses
        similar_count = 0
        for i in range(len(bot_messages) - 1):
            words_i = set(bot_messages[i].split())
            words_next = set(bot_messages[i + 1].split())
            if len(words_i) > 3 and len(words_next) > 3:
                overlap = len(words_i.intersection(words_next))
                similarity = overlap / max(len(words_i), len(words_next))
                if similarity > 0.5:
                    similar_count += 1

        if similar_count >= 2:
            state_notes.append(
                "⚠️ WARNING: Your recent responses are similar - CHANGE APPROACH or TOPIC"
            )

    if state_notes:
        return "\n".join(state_notes)
    return None


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model responses and print them to console for debugging"""
    # Find all think tags
    think_matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)

    # Print think content to console for troubleshooting
    if think_matches:
        print("\n" + "=" * 80)
        print("[THINK TAG DETECTED] Model's internal reasoning:")
        print("=" * 80)
        for i, think_content in enumerate(think_matches, 1):
            print(f"\n--- Think Block {i} ---")
            print(think_content.strip())
            print("--- End Think Block ---")
        print("=" * 80 + "\n")

    # Remove think tags from response
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def generate_intent(context):
    """Use LangChain to identify user intent (internet search, standard chat, etc.)"""
    print(
        f"\n[INTENT DETECTION] Starting intent classification for: '{context[:100]}...'"
    )

    def _run_intent_chain():
        # Create the prompt template for intent classification
        intent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intent classifier for a Discord bot. Your job is to classify messages into exactly one category.

CLASSIFICATION RULES:

Choose 'show-images' if the message asks to SEE, SHOW, or DISPLAY something visual:
- "Show me...", "Can you show...", "Display..."
- "What does X look like?"
- Requests for diagrams, pictures, images, photos
- "I want to see...", "Let me see..."
- Visual references to connectors, components, topologies, etc.

Choose 'internet' ONLY if the message EXPLICITLY asks about:
- Current events, breaking news, today's/recent happenings
- Weather forecasts or current weather conditions
- Real-time data (stock prices, sports scores, cryptocurrency)
- Factual lookups that require up-to-date information
- "Search for...", "Look up...", "What's happening with..."

Choose 'standard' for ALL other messages including:
- Creative writing or storytelling requests
- Personal conversations, feelings, opinions
- Hypothetical questions ("what if...", "imagine...")
- Questions about the bot itself
- General knowledge that doesn't change frequently
- Casual chat
- Training/instruction questions about documentation
- MEMORY/RECALL QUESTIONS - "remember when...", "what do you remember about...", etc.

CRITICAL: Memory, recall, and training questions are ALWAYS 'standard'!

Respond with ONLY one word: show-images OR internet OR standard""",
                ),
                ("human", "{message}"),
            ]
        )

        # Create the chain
        chain = intent_prompt | llm_for_agents | StrOutputParser()

        # Run the chain
        result = chain.invoke({"message": context})
        result = result.strip().lower()

        # Ensure valid output
        if result not in ["internet", "standard", "show-images"]:
            result = "standard"

        print(f"[INTENT DETECTION] Result: {result}")
        return result

    # Run in thread to avoid blocking Discord event loop
    return await asyncio.to_thread(_run_intent_chain)


async def extract_user_facts(
    username: str, message: str, response: str, activity_info: str = None
):
    """Extract new facts about the user from their message, bot response, and activity data using LangChain"""
    print(f"\n[FACT EXTRACTION] Starting fact extraction for user: {username}")
    print(f"[FACT EXTRACTION] Message: '{message[:80]}...'")
    if activity_info:
        print(f"[FACT EXTRACTION] Activity info: {activity_info}")

    def _run_fact_extraction():
        # Build conversation context including activity
        conversation_context = f"""USER ({username}): {message}
{modelName}: {response}"""

        if activity_info:
            conversation_context = f"""USER ACTIVITY:
{activity_info}

CONVERSATION:
{conversation_context}

IMPORTANT: If the user is listening to music, playing a game, or streaming, extract the SPECIFIC details (song name, artist, game name, etc.) as facts."""

        # Create the prompt template for fact extraction
        fact_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a professional memory analyst. Your job is to extract clear, useful facts from conversations.

CRITICAL ATTRIBUTION RULES:
You are analyzing a conversation between {username} and {modelName} (the assistant).

EXTRACT FACTS about what {username} says about THEMSELVES:
- Professional information (job, role, company)
- Technical skills and expertise
- Preferences and interests
- Background and experience

NEVER extract what {modelName} says about itself.
If you're unsure who said something, DO NOT extract it.

INTERPRETATION EXAMPLES:
{username}: "I work at Google" → "works at Google"
{username}: "I'm a Python developer" → "is a Python developer"
{username}: "I study computer science at MIT" → "studies computer science at MIT"
{username}: "I love gaming" → "enjoys video games"

EXTRACT THESE (from {username}'s statements only):
✓ Personal info: location, job, education, role
✓ Technical skills and expertise
✓ Hobbies, interests, skills
✓ Preferences: likes, dislikes, favorites
✓ Professional background: company, team, projects
✓ Goals and plans

DO NOT EXTRACT:
✗ Greetings, commands, meta-conversation
✗ Temporary emotions ("is happy right now")
✗ What {modelName} says about itself

RESPONSE FORMAT:
If you found facts:
FACTS:
- [clear fact about {username}]
- [another clear fact]

If nothing meaningful:
NONE""",
                ),
                ("human", "{conversation}"),
            ]
        )

        # Create the chain
        chain = fact_prompt | llm_for_memory | StrOutputParser()

        # Run the chain
        extraction_result = chain.invoke({"conversation": conversation_context})
        extraction_result = extraction_result.strip()

        print(f"[FACT EXTRACTION] Agent output: {extraction_result}")

        if extraction_result.startswith("FACTS:"):
            facts_text = extraction_result[6:].strip()
            if facts_text:
                # Split by newlines and filter for lines starting with -
                facts_list = []
                for line in facts_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        fact = line[1:].strip()  # Remove the leading dash
                        if fact:
                            facts_list.append(fact)

                print(
                    f"[FACT EXTRACTION] Extracted {len(facts_list)} facts: {facts_list}"
                )
                return facts_list

        print(f"[FACT EXTRACTION] No facts extracted (result was NONE or invalid)")
        return []

    try:
        # Run in thread to avoid blocking Discord event loop
        return await asyncio.to_thread(_run_fact_extraction)
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return []


async def prepare_context_brief(
    messages: List[Dict], current_user: str, current_message: str
) -> dict:
    """
    Prepare a lightweight context summary for the response model using LangChain.
    Focused on training/instruction context - what the user needs help with.

    Returns: A brief contextual summary with:
    - What topic/subject is being discussed
    - User's apparent experience level or role
    - What has already been covered
    - Current question or need
    """
    if not messages or len(messages) < 2:
        return None

    print(
        f"\n[CONTEXT BRIEF] Preparing lightweight context for {len(messages)} messages"
    )

    def _run_context_preparation():
        # Format messages with metadata
        formatted_messages = []

        for msg in messages:
            timestamp = msg.get("timestamp", "unknown")
            username = msg.get("username", "Unknown")
            content = msg.get("content", "")
            is_bot_msg = username == modelName or msg.get("is_bot", False)

            msg_line = f"[{timestamp}] {username}: {content}"
            if is_bot_msg:
                msg_line += f" [{modelName.upper()}'S RESPONSE]"

            formatted_messages.append(msg_line)

        conversation_text = "\n".join(formatted_messages)

        # Create the prompt template for context analysis
        context_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You analyze training/instruction conversations to provide context for the response model.
Your role is to summarize the learning context - NOT to instruct the model on what to do.

Your ONLY job is to clarify:
1. TOPIC: What subject/topic is being discussed or taught
2. PROGRESS: What concepts have already been covered in this conversation
3. USER CONTEXT: Any indication of the user's role, experience level, or background
4. CURRENT NEED: What the user is currently asking about or struggling with

DO NOT:
- Tell {modelName} what to do/say
- Give directives or suggestions
- Make assumptions about what the user should learn next

ATTRIBUTION RULES:
- Messages marked [{modelName.upper()}'S RESPONSE] = what {modelName} explained
- Messages WITHOUT that tag = what the user said/asked
- Be clear about what was taught vs what was asked

Write a MINIMAL context brief (max 300 chars) covering:
- Current topic being discussed
- What's been covered so far
- User's apparent level/role (if evident)
- What they're currently asking about

KEEP IT FACTUAL AND CONCISE.""",
                ),
                (
                    "human",
                    """CONVERSATION HISTORY:
{conversation}

CURRENT MESSAGE {bot_name} needs to respond to:
{user}: {message}""",
                ),
            ]
        )

        # Create the chain
        chain = context_prompt | llm_for_memory | StrOutputParser()

        # Run the chain
        brief = chain.invoke(
            {
                "conversation": conversation_text,
                "bot_name": modelName,
                "user": current_user,
                "message": current_message,
            }
        )
        brief = brief.strip()

        print(f"[CONTEXT BRIEF] Generated {len(brief)} chars of observational context")
        print(f"[CONTEXT BRIEF] Preview:\n{brief[:200]}...")

        return brief

    try:
        return await asyncio.to_thread(_run_context_preparation)
    except Exception as e:
        print(f"Error preparing context brief: {e}")
        return None


async def summarize_channel_activity(
    messages: List[Dict], previous_summary: str = None
) -> str:
    """
    Summarize recent channel conversation using LangChain.
    Creates a comprehensive summary that builds on previous context.
    This summary is used to provide context about WHO is talking TO WHOM about WHAT,
    including relationship dynamics, ongoing topics, and emotional context.
    """
    if not messages:
        return None

    print(
        f"\n[SUMMARIZATION] Starting channel activity summarization for {len(messages)} messages"
    )
    if previous_summary:
        print(
            f"[SUMMARIZATION] Building on previous summary ({len(previous_summary)} chars)"
        )
    else:
        print(f"[SUMMARIZATION] Creating fresh summary from recent messages only")

    def _run_summarization():
        # Format messages for summarization with rich contextual metadata
        formatted_messages = []
        for msg in messages:
            timestamp = msg.get("timestamp", "unknown time")
            username = msg.get("username", "unknown")
            content = msg.get("content", "")
            msg_line = f"[{timestamp}] {username}: {content}"

            # Add reply context if this is a reply
            if msg.get("reply_to"):
                reply_to = msg["reply_to"]
                msg_line += f' [REPLYING TO {reply_to["username"]}: "{reply_to["content"][:50]}..."]'

            # Add mention context if users were mentioned
            if msg.get("mentioned_users"):
                mentions = ", ".join(msg["mentioned_users"])
                msg_line += f" [MENTIONED: {mentions}]"

            formatted_messages.append(msg_line)

        conversation_text = "\n".join(formatted_messages)

        # Build context-aware summary that preserves continuity
        previous_context = ""
        if previous_summary:
            previous_context = f"""
EARLIER CONVERSATION BACKGROUND (for context only):
{previous_summary}

NOTE: The messages below are the RECENT conversation. Create a fresh summary of these recent messages,
but use the background above to understand the full arc and any ongoing dynamics.
"""

        # Create the prompt template for summarization
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a factual reporter who summarizes conversations objectively. You NEVER confuse who said what.

CRITICAL RULES:

1. BE OBJECTIVE: Report what people ACTUALLY said and did. State facts clearly.

2. PERFECT SPEAKER ATTRIBUTION:
   - Username BEFORE the colon = THE SPEAKER (who said it)
   - "user1: can you help?" = user1 said this TO {modelName}
   - NEVER reverse who said what

3. DIRECTIONAL CLARITY:
   - When someone asks or requests something, be clear about WHO asked WHOM
   - Always use names to avoid confusion

4. ALWAYS SPECIFY WHO SAID WHAT TO WHOM
   - NEVER write "{modelName} replied" - write "{modelName} replied TO user1"

FORMAT: Write a 2-4 sentence FACTUAL summary including:
- Main topics/actions
- WHO said/did what TO WHOM
- Current status of the conversation""",
                ),
                (
                    "human",
                    """{previous_context}
RECENT MESSAGES (chronological order, most recent at bottom):
{conversation}

Provide FACTUAL, OBJECTIVE summary of what happened:""",
                ),
            ]
        )

        # Create the chain
        chain = summary_prompt | llm_for_memory | StrOutputParser()

        # Run the chain
        summary = chain.invoke(
            {"previous_context": previous_context, "conversation": conversation_text}
        )
        summary = summary.strip()

        print(
            f"[SUMMARIZATION] Complete: {len(messages)} messages -> {len(summary)} chars"
        )
        print(f"[SUMMARIZATION] Summary: {summary}")
        return summary

    try:
        # Run in thread to avoid blocking Discord event loop
        return await asyncio.to_thread(_run_summarization)
    except Exception as e:
        print(f"Error summarizing channel activity: {e}")
        return None


async def extract_facts_from_channel(messages: List[Dict]) -> Dict[str, List[str]]:
    """Extract memorable facts about users from channel conversation using LangChain"""
    if not messages or len(messages) < 3:  # Need at least a few messages
        return {}

    print(
        f"\n[CHANNEL FACT EXTRACTION] Starting fact extraction from {len(messages)} channel messages"
    )

    def _run_fact_extraction():
        # Format messages for analysis
        formatted_messages = []
        for msg in messages:
            timestamp = msg.get("timestamp", "unknown time")
            username = msg.get("username", "unknown")
            content = msg.get("content", "")
            msg_line = f"[{timestamp}] {username}: {content}"
            formatted_messages.append(msg_line)

        conversation_text = "\n".join(formatted_messages)

        # Get list of all usernames in the conversation
        usernames_list = list(set([msg["username"] for msg in messages]))
        usernames_str = ", ".join(usernames_list)
        print(f"[CHANNEL FACT EXTRACTION] Users in conversation: {usernames_str}")

        # Create the prompt template for channel fact extraction
        channel_fact_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a professional memory analyst. You analyze group conversations and extract clear, useful facts about users.

CRITICAL ATTRIBUTION RULES:
- "{modelName}" is the assistant - NEVER extract facts about what {modelName} says
- EXTRACT FACTS about what HUMAN USERS say about THEMSELVES:
  - Professional information (job, role, company)
  - Technical skills and expertise
  - Preferences and interests
  - Background and experience
- Each message shows "[timestamp] username: message" - pay careful attention to who said what
- If unsure who said something, DO NOT extract it

INTERPRETATION EXAMPLES:
user1: "I work at Google" → "works at Google"
user1: "I'm a Python developer" → "is a Python developer"
user1: "I study CS at MIT" → "studies computer science at MIT"

EXTRACT THESE (from user's OWN statements):
✓ Personal info: location, job, education, role
✓ Technical skills and expertise
✓ Hobbies, interests, skills
✓ Preferences: likes, dislikes, favorites
✓ Professional background: company, team, projects

DO NOT EXTRACT:
✗ Greetings, commands, casual chat
✗ Temporary emotions
✗ What {modelName} says (it's the assistant!)

RESPONSE FORMAT:
username:
- [clear fact]
- [another clear fact]

OR if nothing meaningful: NONE""",
                ),
                (
                    "human",
                    """CONVERSATION:
{conversation}

USERS IN CONVERSATION: {users}

Extract facts organized by username:""",
                ),
            ]
        )

        # Create the chain
        chain = channel_fact_prompt | llm_for_memory | StrOutputParser()

        # Run the chain
        extraction_result = chain.invoke(
            {"conversation": conversation_text, "users": usernames_str}
        )
        extraction_result = extraction_result.strip()

        print(f"[CHANNEL FACT EXTRACTION] Agent output:\n{extraction_result}")

        # Parse the results
        facts_by_user = {}
        if not extraction_result.upper().startswith("NONE"):
            lines = extraction_result.split("\n")
            current_user = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this is a username line (ends with :)
                if line.endswith(":") and not line.startswith("-"):
                    current_user = line[:-1].strip()
                    facts_by_user[current_user] = []
                # Check if this is a fact line (starts with -)
                elif line.startswith("-") and current_user:
                    fact = line[1:].strip()
                    if fact and len(fact) > 10:  # Ignore very short "facts"
                        facts_by_user[current_user].append(fact)

        # Remove users with no facts
        facts_by_user = {k: v for k, v in facts_by_user.items() if v}

        # CRITICAL: Remove any facts about the bot itself
        bot_names = [modelName, modelName.lower(), "assistant", "Assistant"]
        for bot_name in bot_names:
            if bot_name in facts_by_user:
                print(
                    f"[CHANNEL FACT EXTRACTION] REMOVED {len(facts_by_user[bot_name])} facts about bot '{bot_name}' (bot cannot store facts about itself)"
                )
                del facts_by_user[bot_name]

        if facts_by_user:
            print(
                f"[CHANNEL FACT EXTRACTION] Extracted facts for {len(facts_by_user)} users: {list(facts_by_user.keys())}"
            )
            for user, facts in facts_by_user.items():
                print(f"[CHANNEL FACT EXTRACTION] {user}: {len(facts)} facts")
        else:
            print(f"[CHANNEL FACT EXTRACTION] No facts extracted (NONE or invalid)")

        return facts_by_user

    try:
        # Run in thread to avoid blocking Discord event loop
        return await asyncio.to_thread(_run_fact_extraction)
    except Exception as e:
        print(f"Error extracting facts from channel: {e}")
        return {}


async def extract_event_memories(username: str, messages: List[Dict]) -> List[str]:
    """
    Extract specific event instances from recent conversation using LangChain.
    Events are distinct from facts - they are specific occurrences with timestamps.

    Examples of events:
    - "discussed Python project"
    - "debugged code together"
    - "set up development environment"
    """
    if not messages or len(messages) < 3:
        return []

    print(
        f"\n[EVENT EXTRACTION] Starting event extraction from {len(messages)} messages for {username}"
    )

    def _run_event_extraction():
        # Format messages for analysis
        formatted_messages = []
        for msg in messages:
            timestamp = msg.get("timestamp", "unknown")
            msg_username = msg.get("username", "Unknown")
            content = msg.get("content", "")
            is_bot_msg = msg_username == modelName or msg.get("is_bot", False)

            msg_line = f"[{timestamp}] {msg_username}: {content}"
            if is_bot_msg:
                msg_line += f" [{modelName.upper()}'S RESPONSE]"

            formatted_messages.append(msg_line)

        conversation_text = "\n".join(formatted_messages)

        # Create the prompt template for event extraction
        event_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a precise memory analyst that extracts specific event instances.

CRITICAL: ONLY extract events that ACTUALLY HAPPENED in the conversation.
NEVER hallucinate or infer events that weren't explicitly described.

You distinguish between:
- FACTS: General truths (e.g., "works at Google", "knows Python")
- EVENTS: Specific instances with actions (e.g., "discussed project setup", "debugged code together")

ONLY extract EVENTS - specific things that happened in this conversation.

WHAT IS AN EVENT (ALL must include participants):
✅ "discussed Python project with {modelName}"
✅ "debugged code with {modelName}"
✅ "set up development environment with {modelName}"

WHAT IS NOT AN EVENT:
❌ "knows Python" (fact, not event)
❌ "wants to learn React" (desire, hasn't happened)
❌ "enjoys coding" (preference, not specific instance)

RESPONSE FORMAT:
If you found events that ACTUALLY HAPPENED:
EVENTS:
- [brief event description with action]
- [another event that happened]

If no specific events occurred:
NONE

Be STRICT. Only extract events that actually occurred.""",
                ),
                (
                    "human",
                    """CONVERSATION:
{conversation}

Extract specific EVENT INSTANCES that occurred between {user} and {bot_name}:""",
                ),
            ]
        )

        # Create the chain
        chain = event_prompt | llm_for_memory | StrOutputParser()

        # Run the chain
        extraction_result = chain.invoke(
            {"conversation": conversation_text, "user": username, "bot_name": modelName}
        )
        extraction_result = extraction_result.strip()

        print(f"[EVENT EXTRACTION] Agent output: {extraction_result}")

        if extraction_result.startswith("EVENTS:"):
            events_text = extraction_result[7:].strip()
            if events_text:
                events_list = []
                for line in events_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        event = line[1:].strip()
                        if event and len(event) > 5:
                            events_list.append(event)

                print(
                    f"[EVENT EXTRACTION] Extracted {len(events_list)} events: {events_list}"
                )
                return events_list

        print(f"[EVENT EXTRACTION] No events extracted (result was NONE or invalid)")
        return []

    try:
        return await asyncio.to_thread(_run_event_extraction)
    except Exception as e:
        print(f"Error extracting events: {e}")
        return []


async def get_context_with_memories(username: str, message: str):
    """
    Get LONG-TERM memory context for the user including documentation retrieval.
    Documentation has higher priority than personal user facts for training/instruction purposes.
    Since we use single-turn conversation, this needs to provide enough context
    for the bot to understand the user and situation.
    """
    context_parts = []

    context_parts.append(f"Current user: {username}")

    # PRIORITY 1: Retrieve relevant documentation (higher priority than personal facts)
    relevant_docs = retrieve_relevant_docs(message, k=3)
    if relevant_docs:
        context_parts.append("\nTRAINING DOCUMENTATION:")
        for i, doc in enumerate(relevant_docs, 1):
            content = doc["content"]
            source = doc["source"]
            page = doc["page"]

            # Truncate long docs to keep context manageable
            content_preview = content[:500] + "..." if len(content) > 500 else content

            # Format source citation
            if page is not None:
                citation = f"[Source: {source}, Page {page + 1}]"
            else:
                citation = f"[Source: {source}]"

            context_parts.append(f"{citation}:\n{content_preview}")
        context_parts.append("")  # Add spacing

    # Get relevant facts, events, and tasks - expanded for better context
    relevant_facts = user_memory.get_relevant_facts(username, message, max_facts=3)
    relevant_events = user_memory.get_relevant_events(username, message, max_events=3)
    relevant_tasks = user_memory.get_relevant_tasks(username, message, max_tasks=2)
    other_person_facts = None
    other_person_events = None
    other_person_name = None
    message_lower = message.lower()

    # Check if the message is asking about someone else
    for user_key in user_memory.memories.keys():
        # Check both the username and aliases
        names_to_check = [user_key] + user_memory.memories[user_key].get("aliases", [])
        for name in names_to_check:
            if name.lower() in message_lower and name.lower() != username.lower():
                # They're asking about this person
                other_person_name = user_key
                other_person_facts = user_memory.get_user_facts(user_key)
                other_person_events = user_memory.get_user_events(
                    user_key, max_events=5
                )
                break
        if other_person_name:
            break

    # Add relevant facts if any (only highly relevant ones)
    if relevant_facts:
        context_parts.append("Relevant facts about this user:")
        for fact in relevant_facts:
            context_parts.append(f"- {fact}")

    # Add facts about the OTHER person if they're being asked about
    if other_person_facts and other_person_name:
        context_parts.append(f"\nRelevant facts about {other_person_name}:")
        for fact in other_person_facts[:5]:  # Limit to 5 most important facts
            context_parts.append(f"- {fact}")

    # Add events about the OTHER person if they're being asked about
    if other_person_events and other_person_name:
        context_parts.append(f"\nRelevant past events involving {other_person_name}:")
        for event in other_person_events:
            event_desc = event.get("description", "")
            timestamp = event.get("timestamp", "")
            participants = event.get("participants", [])
            participants_str = ", ".join(participants) if participants else "unknown"
            context_parts.append(
                f"- [{timestamp}] {event_desc} (involving: {participants_str})"
            )

    # Add relevant event memories (specific instances that occurred)
    if relevant_events:
        context_parts.append("Relevant past events:")
        for event in relevant_events:
            event_desc = event.get("description", "")
            timestamp = event.get("timestamp", "")
            participants = event.get("participants", [])

            # Add clear attribution - show who the event is about
            # This prevents confusion when retrieving events about other users
            participants_str = ", ".join(participants) if participants else "unknown"
            context_parts.append(
                f"- [{timestamp}] {event_desc} (involving: {participants_str})"
            )

    # Add relevant past tasks (only most recent/relevant)
    if relevant_tasks:
        context_parts.append("Relevant past tasks:")
        for task in relevant_tasks:
            task_type = task.get("type", "unknown")
            request = (
                task.get("request", "")[:80] + "..."
                if len(task.get("request", "")) > 80
                else task.get("request", "")
            )
            outcome = (
                task.get("outcome", "")[:100] + "..."
                if len(task.get("outcome", "")) > 100
                else task.get("outcome", "")
            )
            timestamp = task.get("timestamp", "")
            context_parts.append(
                f"- [{task_type}] {timestamp}: requested '{request}' -> {outcome}"
            )

    # DO NOT include recent conversation here - it's already in the chatlog!
    # The chatlog provides immediate/short-term context
    # This memory context is for LONG-TERM persistent information only

    return "\n".join(context_parts)


async def searchweb(query):
    """
    Search the web using DuckDuckGo and return results with actual webpage content.

    Process:
    1. Try DuckDuckGo instant answer API for quick facts
    2. If no instant answer, scrape actual web pages for detailed content
    3. Return formatted results with sources
    """
    try:
        # Step 1: Try DuckDuckGo instant answer API first (fast, structured data)
        search_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1&skip_disambig=1"

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(search_url, headers=headers, timeout=10)
        data = response.json()

        results = []

        # Try different instant answer types in priority order
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
            if data.get("AbstractSource"):
                results.append(f"Source: {data['AbstractSource']}")
            return "\n".join(results)

        # Get definition if available
        if data.get("Definition"):
            results.append(f"Definition: {data['Definition']}")
            if data.get("DefinitionSource"):
                results.append(f"Source: {data['DefinitionSource']}")
            return "\n".join(results)

        # Get answer if available
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")
            if data.get("AnswerType"):
                results.append(f"Type: {data['AnswerType']}")
            return "\n".join(results)

        # Step 2: If no instant answers, search and scrape actual web pages
        print("No instant answers found, searching web pages...")

        # Use DuckDuckGo Lite for simple HTML parsing
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        response = requests.get(search_url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract URLs from search results
            links = soup.find_all("a")
            web_urls = []

            for link in links:
                href = link.get("href")
                if href and href.startswith("http") and "duckduckgo" not in href:
                    # Avoid social media and problematic sites
                    if not any(
                        blocked in href.lower()
                        for blocked in [
                            "facebook",
                            "twitter",
                            "instagram",
                            "tiktok",
                            "youtube",
                        ]
                    ):
                        web_urls.append(href)
                        if len(web_urls) >= 3:  # Limit to 3 websites to scrape
                            break

            if web_urls:
                scraped_content = []
                for i, url in enumerate(web_urls, 1):
                    try:
                        print(f"Scraping website {i}: {url[:50]}...")
                        content = await scrape_webpage_content(url, query)
                        if content:
                            scraped_content.append(f"From {url[:30]}...\n{content}")

                        # Avoid being too aggressive
                        if i < len(web_urls):
                            await asyncio.sleep(1)

                    except Exception as e:
                        print(f"Failed to scrape {url}: {e}")
                        continue

                if scraped_content:
                    return f"Web search results for: {query}\n\n" + "\n\n---\n\n".join(
                        scraped_content
                    )

        return f"Search completed for '{query}' but no detailed information could be retrieved."

    except requests.exceptions.Timeout:
        return "Search request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Search error: Unable to connect to search service"
    except Exception as e:
        print(f"Search error: {e}")
        return "An error occurred while searching. Please try again."


async def search_web_images(query: str, max_images: int = 2) -> List[str]:
    """
    Search for images on the web related to the query.

    Returns a list of image file paths that were downloaded.
    """
    import json
    import os
    import tempfile

    downloaded_images = []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Use Bing image search
        bing_url = f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2&first=1"

        print(f"[WEB IMAGES] Searching Bing for: {query}")
        response = requests.get(bing_url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            image_urls = []

            # Primary method: Extract from iusc anchor tags with JSON metadata
            for a_tag in soup.find_all("a", class_="iusc"):
                m_attr = a_tag.get("m")
                if m_attr:
                    try:
                        m_data = json.loads(m_attr)
                        if "murl" in m_data:
                            img_url = m_data["murl"]
                            # Skip data URLs and very short URLs
                            if img_url.startswith("http") and len(img_url) > 20:
                                image_urls.append(img_url)
                                print(
                                    f"[WEB IMAGES] Found image URL: {img_url[:80]}..."
                                )
                                if len(image_urls) >= max_images:
                                    break
                    except json.JSONDecodeError:
                        continue

            # Fallback: Try to find thumbnail images and get their source
            if not image_urls:
                print("[WEB IMAGES] Trying fallback method...")
                for img in soup.find_all("img"):
                    src = img.get("src2") or img.get("data-src") or img.get("src")
                    if src and src.startswith("http"):
                        # Skip Bing's own assets
                        if "bing.com" not in src and "microsoft.com" not in src:
                            image_urls.append(src)
                            if len(image_urls) >= max_images:
                                break

            print(f"[WEB IMAGES] Found {len(image_urls)} image URLs")

            # Download images
            for i, img_url in enumerate(image_urls):
                try:
                    print(f"[WEB IMAGES] Downloading image {i + 1}: {img_url[:60]}...")

                    img_response = requests.get(img_url, headers=headers, timeout=10)
                    if img_response.status_code == 200:
                        # Check content length to avoid tiny images
                        content_length = len(img_response.content)
                        if content_length < 5000:
                            print(
                                f"[WEB IMAGES] Skipping small image ({content_length} bytes)"
                            )
                            continue

                        # Determine file extension
                        content_type = img_response.headers.get("content-type", "")
                        if "png" in content_type:
                            ext = "png"
                        elif "gif" in content_type:
                            ext = "gif"
                        elif "webp" in content_type:
                            ext = "webp"
                        else:
                            ext = "jpg"

                        # Save to temp file
                        temp_path = os.path.join(
                            tempfile.gettempdir(),
                            f"web_image_{i}_{hashlib.md5(img_url.encode()).hexdigest()[:8]}.{ext}",
                        )

                        with open(temp_path, "wb") as f:
                            f.write(img_response.content)

                        downloaded_images.append(temp_path)
                        print(
                            f"[WEB IMAGES] Saved image to {temp_path} ({content_length} bytes)"
                        )

                except Exception as e:
                    print(f"[WEB IMAGES] Failed to download image: {e}")
                    continue

        return downloaded_images

    except Exception as e:
        print(f"[WEB IMAGES] Search error: {e}")
        return []


async def scrape_webpage_content(url: str, query: str, max_length: int = 2000):
    """Scrape content from a webpage and extract relevant information"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Try to find main content areas
        main_content = None
        for selector in ["main", "article", ".content", "#content", ".main", "#main"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find("body")

        if main_content:
            # Extract text
            text = main_content.get_text(separator=" ", strip=True)

            # Clean up text
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            clean_text = " ".join(lines)

            # If text is too long, try to find the most relevant parts
            if len(clean_text) > max_length:
                query_words = query.lower().split()

                # Split into sentences and score relevance
                sentences = [s.strip() for s in clean_text.split(".") if s.strip()]
                scored_sentences = []

                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    score = sum(1 for word in query_words if word in sentence_lower)
                    if score > 0:
                        scored_sentences.append((sentence, score))

                # Sort by relevance and take top sentences
                if scored_sentences:
                    scored_sentences.sort(key=lambda x: x[1], reverse=True)
                    relevant_text = ". ".join([s[0] for s in scored_sentences[:5]])
                    if len(relevant_text) > max_length:
                        relevant_text = relevant_text[:max_length] + "..."
                    return relevant_text
                else:
                    # If no query-specific content, just take the beginning
                    return clean_text[:max_length] + "..."

            return clean_text

        return None

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


# ============================================================================
# BOT EVENTS
# ============================================================================
@client.event
async def on_ready():
    """
    Called when bot successfully connects to Discord.
    Initializes slash commands and documentation system.
    """
    print(f"Successfully logged in as {client.user}.")

    # Sync slash commands with Discord first (fast)
    try:
        synced = await client.tree.sync()
        print(f"Synced {len(synced)} command/s.")
    except Exception as e:
        print(e)

    # Initialize documentation retrieval system in background thread
    # This prevents blocking Discord's heartbeat for large documents
    try:
        print(f"[DOCS] Starting document indexing in background...")
        doc_count = await asyncio.to_thread(load_training_documents)
        if doc_count > 0:
            print(f"[DOCS] Initialized with {doc_count} document chunks")
        else:
            print(f"[DOCS] No training documents found. Add files to {DOCS_DIRECTORY}")
    except Exception as e:
        print(f"[DOCS] Error initializing documentation system: {e}")

    # Initialize PDF image extraction in background thread
    try:
        print(f"[IMAGES] Starting image extraction in background...")
        image_count = await asyncio.to_thread(load_pdf_images)
        if image_count > 0:
            print(f"[IMAGES] Initialized with {image_count} images")
        else:
            print(f"[IMAGES] No images extracted from PDFs")
    except Exception as e:
        print(f"[IMAGES] Error initializing image system: {e}")


@client.event
async def on_message(message):
    """
    Handle incoming messages and route to appropriate LLM processing.
    Supports both text-only and image+text messages in DMs and group channels.
    Tracks all group channel messages for context awareness.
    """
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # ========== TRACK GROUP CHANNEL MESSAGES ==========
    # Track all messages in group channels (not DMs) for context with rich metadata
    if not isinstance(message.channel, discord.DMChannel):
        # Enrich message with contextual metadata
        enriched_msg = await enrich_message_context(message)

        channel_history.add_message(
            channel_id=message.channel.id,
            username=enriched_msg["username"],
            content=enriched_msg["content"],
            timestamp=enriched_msg["timestamp"],
            metadata=enriched_msg,
        )

        # Increment bot response cooldown counter (for anti-loop protection)
        if not message.author.bot:
            bot_interaction_tracking["messages_since_bot_response"] += 1

    # ========== GROUP CHANNEL MESSAGES (when bot is mentioned or replied to) ==========
    # Check if bot was @mentioned (Discord mention only, NOT just name in text)
    # Name in text is handled by proactive triggers system below
    is_bot_mentioned = client.user in message.mentions
    is_reply_to_bot = (
        message.reference
        and message.reference.resolved
        and message.reference.resolved.author == client.user
    )

    # Check if this is another bot's message (not self, but is a bot)
    is_other_bot = message.author.bot and message.author != client.user

    if (is_bot_mentioned or is_reply_to_bot) and not isinstance(
        message.channel, discord.DMChannel
    ):
        # ========== ANTI-LOOP PROTECTION ==========
        # Check if we should respond based on anti-loop rules
        should_respond = True

        # Rule 1: Never respond if last 2 messages were from bots (require human message)
        recent_msgs = channel_history.get_recent_messages(message.channel.id, limit=3)
        if len(recent_msgs) >= 2:
            last_two_bots = all(msg.get("is_bot", False) for msg in recent_msgs[-2:])
            if last_two_bots and message.author.bot:
                print(
                    f"[ANTI-LOOP] Blocked: Last 2 messages were from bots, need human message"
                )
                should_respond = False

        # Rule 2: If this is from another bot, check if we just responded to them
        if is_other_bot and should_respond:
            if (
                bot_interaction_tracking["last_bot_responded_to_id"]
                == message.author.id
                and bot_interaction_tracking["messages_since_bot_response"] < 3
            ):
                print(
                    f"[ANTI-LOOP] Blocked: Just responded to {message.author.display_name}, cooldown active"
                )
                should_respond = False

        # Rule 3: ALWAYS respond to direct mentions/replies (overrides cooldown)
        if is_bot_mentioned or is_reply_to_bot:
            should_respond = True
            print(
                f"[BOT-TO-BOT] Direct mention/reply detected, responding despite cooldown"
            )

        if not should_respond:
            return

        trigger_type = "replied to" if is_reply_to_bot else "mentioned"
        print(f"\n{'=' * 80}")
        print(
            f"[BOT {trigger_type.upper()}] Bot {trigger_type} in channel by {message.author.display_name}"
        )
        print(f"[BOT {trigger_type.upper()}] Message: '{message.content}'")
        print(f"{'=' * 80}\n")

        # Update tracking if responding to another bot
        if is_other_bot:
            bot_interaction_tracking["last_bot_responded_to_id"] = message.author.id
            bot_interaction_tracking["last_bot_responded_to_timestamp"] = (
                message.created_at
            )
            bot_interaction_tracking["messages_since_bot_response"] = 0
            print(
                f"[BOT-TO-BOT] Updated tracking: responded to bot {message.author.display_name}"
            )

        async with message.channel.typing():
            # Fetch recent channel history if we don't have any context yet
            recent_messages = await fetch_recent_channel_messages(
                message.channel, limit=30
            )

            print(f"[CONTEXT] Got {len(recent_messages)} messages from channel history")

            # Generate contextual summary for emotional tone and context understanding
            if len(recent_messages) > 3:
                print(
                    f"[CONTEXT] Starting contextual analysis of {len(recent_messages)} messages..."
                )
                recent_for_summary = (
                    recent_messages[-50:]
                    if len(recent_messages) > 50
                    else recent_messages
                )

                # Get previous summary for continuity
                previous_summary = channel_history.get_summary(message.channel.id)

                # Generate summary focused on tone, emotion, and context (not exact words)
                summary = await summarize_channel_activity(
                    recent_for_summary, previous_summary
                )
                if summary:
                    channel_history.set_summary(message.channel.id, summary)
                    print(f"[CONTEXT] Contextual summary stored for channel")

                # Extract facts periodically (not every single message to avoid redundancy)
                if channel_history.should_extract_facts(
                    message.channel.id, message_threshold=20
                ):
                    print(
                        f"[CONTEXT] Starting fact extraction from channel messages..."
                    )
                    facts_by_user = await extract_facts_from_channel(recent_for_summary)
                    for username, facts in facts_by_user.items():
                        for fact in facts:
                            if user_memory.add_user_fact(username, fact):
                                print(f"[MEMORY] Stored fact: {username} - {fact}")
                    channel_history.mark_facts_extracted(message.channel.id)
                    print(f"[CONTEXT] Fact extraction complete, marked checkpoint")
                else:
                    print(
                        f"[CONTEXT] Skipping fact extraction (not enough new messages since last extraction)"
                    )

                # Don't clear messages - let max_messages limit handle trimming
                # This preserves context for future reference

            # Handle image attachments
            if message.attachments:
                await _process_image_message(
                    message, None, is_dm=False, channel_id=message.channel.id
                )
            # Handle text-only messages
            else:
                await _process_text_message(
                    message, None, is_dm=False, channel_id=message.channel.id
                )

    # ========== DIRECT MESSAGES ==========
    elif isinstance(message.channel, discord.DMChannel):
        async with message.channel.typing():
            # Handle image attachments
            if message.attachments:
                await _process_image_message(
                    message, chatlogDM, is_dm=True, channel_id=None
                )
            # Handle text-only messages
            else:
                await _process_text_message(
                    message, chatlogDM, is_dm=True, channel_id=None
                )


async def _process_image_message(message, chatlog, is_dm, channel_id=None):
    """
    Process messages containing image attachments using LLaVA vision model.

    Args:
        message: Discord message object with attachment
        chatlog: Chat history (used for DMs only)
        is_dm: Boolean indicating if this is a direct message
        channel_id: Channel ID for group messages (None for DMs)
    """
    # Download and save the image attachment locally
    attachment = message.attachments[0]
    await attachment.save(f"{attachment.filename}")
    loadedimg = f"{attachment.filename}"
    print("image found, saving...")

    username = message.author.display_name

    # Prepare prompt with author name
    modprompt = f"{username}: {message.content}"

    # Build context from appropriate source
    if is_dm:
        # For DMs: Use chatlogDM and add this message
        feed_dict = {"role": "user", "content": f"{username}: {message.content}"}
        chatlog.append(feed_dict)
        context_messages = chatlog[-16:] if len(chatlog) > 16 else chatlog
    else:
        # For group channels: Build from channel_history
        context_messages = build_chatlog_from_channel_history(channel_id, limit=40)
        print(
            f"[IMAGE] Built context from channel_history: {len(context_messages)} messages"
        )

    # Generate AI response using image model (run in thread to avoid blocking)
    # Note: llava doesn't use chat history, just the prompt + image
    ai_response = await asyncio.to_thread(
        ollama.generate, model=imageModel, prompt=modprompt, images=[loadedimg]
    )
    resp = strip_think_tags(ai_response["response"])

    # For DMs, add response to chatlogDM
    if is_dm:
        resp_dict = {"role": "assistant", "content": resp}
        chatlog.append(resp_dict)

    # Send response to user (use reply for images since they're direct responses)
    await send_long_message(message, resp)

    # Add bot response to channel history (group channels only)
    if not is_dm:
        channel_history.add_message(
            channel_id=channel_id,
            username=modelName,
            content=resp,
            metadata={
                "username": modelName,
                "content": resp,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "is_bot": True,
                "author_id": client.user.id,
            },
        )


async def _process_text_message(message, chatlog, is_dm, channel_id=None):
    """
    Process text-only messages with intent detection and memory integration.

    Process flow:
    1. Clean and prepare the message
    2. Detect intent (internet search vs standard chat)
    3. Route to appropriate handler:
       - Internet: Web search + AI-filtered results
       - Standard: Conversation with long-term memory context
    4. Extract and store new facts about the user
    5. Send response and update history

    Args:
        message: Discord message object
        chatlog: Chat history (used for DMs only)
        is_dm: Boolean indicating if this is a direct message
        channel_id: Channel ID for group messages (None for DMs)
    """
    username = message.author.display_name

    print(f"\n[MESSAGE PROCESSING] Processing text message from {username}")
    print(f"[MESSAGE PROCESSING] Message type: {'DM' if is_dm else 'Group Channel'}")

    # Remove bot mentions from the message content
    full_string = message.content
    cut_string = re.sub("<.*?>", modelName, full_string)
    print(f"[MESSAGE PROCESSING] Cleaned message: '{cut_string[:100]}...'")

    # For DMs, add user message to chatlogDM
    # For group channels, messages are already in channel_history
    if is_dm:
        prompt_dict = {"role": "user", "content": f"{username}: {cut_string}"}
        chatlog.append(prompt_dict)

    # Detect intent
    msg_intent = await generate_intent(cut_string)
    print(f"[MESSAGE PROCESSING] Intent: {msg_intent}")

    if msg_intent == "internet":
        # ========== WEB SEARCH ==========
        print(f"\n[WEB SEARCH] Starting web search for: '{cut_string}'")
        search_resp = await searchweb(cut_string)
        print(f"[WEB SEARCH] Got {len(search_resp)} chars of results")
        print(f"[WEB SEARCH] Preview: {search_resp[:200]}...")

        # Get user context for personalized response (long-term memory only)
        memory_context = await get_context_with_memories(username, cut_string)
        print(f"[WEB SEARCH] Memory context prepared ({len(memory_context)} chars)")

        # Create enhanced context for the filter with user memories
        context_enhanced_search = f"Search results: {search_resp}\n\nUser context: {memory_context}\nOriginal request: {cut_string}"

        filter_dict = {"role": "user", "content": context_enhanced_search}
        filterlog.clear()
        filterlog.append(filter_dict)

        # Generate response based on search results (run in thread to avoid blocking)
        print(f"[WEB SEARCH] Generating LLM response from search results...")
        if USE_OPENROUTER:
            ai_response = await asyncio.to_thread(
                openrouter_chat, model=myModel, messages=filterlog
            )
        else:
            ai_response = await asyncio.to_thread(
                ollama.chat, model=myModel, messages=filterlog
            )
        resp = strip_think_tags(ai_response["message"]["content"])

        # For DMs, add response to chatlogDM
        if is_dm:
            resp_dict = {"role": "assistant", "content": resp}
            chatlog.append(resp_dict)

        # Record task memory
        outcome_summary = (
            f"Searched web successfully, found information: {search_resp[:150]}..."
        )
        user_memory.add_task_memory(username, "internet", cut_string, outcome_summary)
        print(f"[WEB SEARCH] Recorded task memory for {username}")
        print(f"[WEB SEARCH] Response length: {len(resp)} chars")

    elif msg_intent == "show-images":
        # ========== SHOW IMAGES ==========
        print(f"\n[SHOW IMAGES] Processing image request for: '{cut_string}'")

        # Get recent conversation to understand what topic the user wants images of
        if channel_id:
            recent_msgs = channel_history.get_recent_messages(channel_id, limit=5)
            # Build conversation context to extract topic
            conv_context = "\n".join(
                [
                    f"{m.get('username', 'User')}: {m.get('content', '')}"
                    for m in recent_msgs
                ]
            )
        else:
            conv_context = cut_string

        # Use LLM to extract the actual topic they want images of
        topic_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract the specific topic the user wants to see images/diagrams of from this conversation. Return ONLY the topic in 2-5 words, nothing else. Example: 'VPN tunnel diagram' or 'fiber optic connectors'",
                ),
                (
                    "human",
                    f"Conversation:\n{conv_context}\n\nUser request: {cut_string}",
                ),
            ]
        )
        topic_chain = topic_prompt | llm_for_agents | StrOutputParser()
        image_search_topic = await asyncio.to_thread(
            topic_chain.invoke, {"conversation": conv_context, "request": cut_string}
        )
        image_search_topic = image_search_topic.strip()
        print(f"[SHOW IMAGES] Extracted topic for image search: '{image_search_topic}'")

        # Get relevant documentation for the extracted topic
        memory_context = await get_context_with_memories(username, image_search_topic)
        print(f"[SHOW IMAGES] Memory context prepared ({len(memory_context)} chars)")

        # For show-images, just provide a brief intro - the images speak for themselves
        resp = f"Here are diagrams related to {image_search_topic}:"

        # For DMs, add response to chatlogDM
        if is_dm:
            resp_dict = {"role": "assistant", "content": resp}
            chatlog.append(resp_dict)

        # Store the topic for image retrieval later
        cut_string = image_search_topic  # Replace with extracted topic for image search

        print(f"[SHOW IMAGES] Response length: {len(resp)} chars")

    else:
        # ========== STANDARD CONVERSATION ==========
        print(f"\n[STANDARD CHAT] Processing standard conversation")

        # Get long-term memory context (facts, events, past tasks)
        memory_context = await get_context_with_memories(username, cut_string)
        print(
            f"[STANDARD CHAT] Long-term memory context prepared ({len(memory_context)} chars)"
        )

        # ========== BUILD MESSAGES FOR LLM ==========
        # CRITICAL: Agents handle all memory/summarization, so we ONLY send:
        # 1. The current user message
        # 2. The context (summaries, facts, etc.) embedded in it
        # We do NOT send the full chatlog - that creates dual context confusion

        # Build a simple single-turn conversation with ONLY the current message
        # NOTE: Don't include username prefix here - it's already in RECENT MESSAGES context
        messages_for_llm = [{"role": "user", "content": cut_string}]
        print(f"[STANDARD CHAT] Using single-turn conversation (agents handle history)")

        # For repetition checking, get recent bot responses from channel_history
        # Fetch more messages to find bot's actual responses (not just recent time window)
        if not is_dm:
            recent_for_repetition = channel_history.get_recent_messages(
                channel_id, limit=60
            )
            recent_bot_responses = [
                msg["content"]
                for msg in recent_for_repetition
                if msg.get("author_id") == client.user.id
            ][-10:]  # Last 10 bot responses only
        else:
            recent_bot_responses = [
                msg["content"] for msg in chatlog[-10:] if msg["role"] == "assistant"
            ]

        print(
            f"[STANDARD CHAT] Tracking {len(recent_bot_responses)} recent bot responses for anti-repetition"
        )

        # ========== SEND TO LLM ==========
        # Context will be embedded in the user message below

        # Build context string - simplified for training assistant
        context_parts = []

        # Add memory context FIRST (documentation + user facts) - this is the priority
        if memory_context and len(memory_context) > 20:
            context_parts.append(memory_context)

        # Initialize activity info variable (used later for fact extraction)
        user_activity_info = None

        # Add recent conversation history
        if is_dm:
            # For DMs, fetch actual Discord message history
            try:
                dm_messages = []
                async for msg in message.channel.history(limit=7):
                    dm_messages.append(
                        {
                            "username": msg.author.display_name,
                            "content": msg.content,
                            "is_bot": msg.author.id == client.user.id,
                        }
                    )

                # Reverse to get chronological order
                dm_messages.reverse()

                if dm_messages:
                    dialogue_lines = []
                    for msg_data in dm_messages:
                        msg_username = (
                            modelName if msg_data["is_bot"] else msg_data["username"]
                        )
                        dialogue_lines.append(f"{msg_username}: {msg_data['content']}")

                    dialogue_history = "\n".join(dialogue_lines)
                    context_parts.append(f"RECENT MESSAGES:\n{dialogue_history}")

            except Exception as e:
                print(f"[DIALOGUE - DM] Error fetching DM history: {e}")
        elif channel_id:
            # Get recent channel messages
            recent_channel_msgs = channel_history.get_recent_messages(
                channel_id, limit=10
            )
            print(
                f"[DIALOGUE] Retrieved {len(recent_channel_msgs)} recent messages from channel history"
            )

            if recent_channel_msgs:
                dialogue_lines = []
                for msg in recent_channel_msgs:
                    msg_username = msg.get("username", "Unknown")
                    msg_content = msg.get("content", "")
                    dialogue_lines.append(f"{msg_username}: {msg_content}")

                dialogue_history = "\n".join(dialogue_lines)
                context_parts.append(f"RECENT MESSAGES:\n{dialogue_history}")

            # Generate attribution context for group channels (important for multiple trainees)
            if len(recent_channel_msgs) > 2:
                context_brief = await prepare_context_brief(
                    recent_channel_msgs, username, cut_string
                )
                if context_brief:
                    context_parts.append(f"ATTRIBUTION:\n{context_brief}")
                    print(
                        f"[ATTRIBUTION] Added context brief ({len(context_brief)} chars)"
                    )

        # Embed combined context into the user message
        if context_parts:
            combined_context = "\n\n".join(context_parts)
            current_msg = messages_for_llm[0]["content"]

            # Embed context BEFORE the message so documentation is seen first
            # Add explicit citation reminder at the end
            messages_for_llm[0]["content"] = (
                f"---\n{combined_context}\n---\n\n"
                f"User message: {current_msg}\n\n"
                f"REMINDER: You MUST cite the source and page number for each fact. "
                f"Example: 'According to the Network+ Guide (Page 378), ...'"
            )
            print(
                f"[LLM] Embedded {len(combined_context)} chars of context into user message"
            )
        else:
            print(f"[LLM] No context to embed")

        # Generate response (with retry on severe repetition)
        print(f"[LLM] Generating response...")

        # Debug: Print full messages being sent to the model
        print("\n" + "=" * 80)
        print("[LLM DEBUG] Full messages sent to model:")
        print("=" * 80)
        for idx, msg in enumerate(messages_for_llm, 1):
            role = msg["role"]
            content = msg["content"]
            print(f"\n--- Message {idx} ({role}) ---")
            print(content[:2000] + "..." if len(content) > 2000 else content)
            print(f"--- End Message {idx} ---")
        print("=" * 80 + "\n")

        # Try generating up to 2 times if we detect severe repetition
        max_retries = 2
        retry_count = 0
        resp = None
        regeneration_triggered = False

        for attempt in range(max_retries):
            if attempt > 0:
                print(
                    f"[LLM] Retry attempt {attempt}/{max_retries - 1} with stronger anti-repetition prompt"
                )
                # Build a list of opening lines to explicitly avoid
                opening_lines_to_avoid = []
                for prev_resp in recent_bot_responses[-3:]:
                    prev_words = prev_resp.lower().strip().split()
                    if len(prev_words) >= 5:
                        opening_lines_to_avoid.append(" ".join(prev_words[:5]))

                avoid_list = "\n".join(
                    [
                        f'- DON\'T START WITH: "{line}"'
                        for line in opening_lines_to_avoid
                    ]
                )

                # Add a strong anti-repetition warning to the user message
                original_content = messages_for_llm[0]["content"]
                messages_for_llm[0]["content"] = f"""{original_content}

CRITICAL REGENERATION WARNING
Your PREVIOUS response had repeated phrases! You MUST generate something COMPLETELY DIFFERENT.

FORBIDDEN OPENING LINES (DO NOT USE ANY OF THESE):
{avoid_list}

START your response with DIFFERENT words. Use a DIFFERENT greeting. Say something NEW!"""
                regeneration_triggered = True

            if USE_OPENROUTER:
                ai_response = await asyncio.to_thread(
                    openrouter_chat, model=myModel, messages=messages_for_llm
                )
            else:
                ai_response = await asyncio.to_thread(
                    ollama.chat, model=myModel, messages=messages_for_llm
                )
            resp = strip_think_tags(ai_response["message"]["content"])
            print(f"[LLM] Response generated ({len(resp)} chars)")

            # Check for internal repetition (same sentence appearing twice within this response)
            sentences = [s.strip() for s in resp.split(".") if len(s.strip()) > 20]
            unique_sentences = set(s.lower() for s in sentences)
            if len(sentences) > len(unique_sentences):
                print(f"[ANTI-REPETITION] ❌ Found duplicate sentence within response")
                print(
                    f"[ANTI-REPETITION] Total sentences: {len(sentences)}, Unique: {len(unique_sentences)}"
                )
                # Trigger retry
                if attempt < max_retries - 1:
                    should_retry = True
                    continue

            # Quick check: Does this response have severe repetition?
            # Only retry if we detect EXACT duplicates or opening line repetition
            should_retry = False
            if recent_bot_responses and attempt < max_retries - 1:
                resp_lower = resp.lower().strip()
                resp_words = resp_lower.split()
                current_opening = (
                    " ".join(resp_words[:5]) if len(resp_words) >= 5 else None
                )

                for prev_resp in recent_bot_responses[-3:]:
                    prev_lower = prev_resp.lower().strip()
                    # Exact duplicate?
                    if resp_lower == prev_lower:
                        print(f"[LLM] Exact duplicate detected, will retry")
                        should_retry = True
                        break
                    # Same opening line?
                    prev_words = prev_lower.split()
                    prev_opening = (
                        " ".join(prev_words[:5]) if len(prev_words) >= 5 else None
                    )
                    if (
                        current_opening
                        and prev_opening
                        and current_opening == prev_opening
                    ):
                        print(f"[LLM] Repeated opening line detected, will retry")
                        should_retry = True
                        break

            if not should_retry:
                print(f"[LLM] Response acceptable, proceeding")
                break

        if regeneration_triggered:
            print(f"[LLM] Completed generation after {attempt + 1} attempts")

        # CRITICAL: Check for exact or near-exact response repetition
        # If the bot already said this EXACT thing or something very similar, reject it and log error
        if recent_bot_responses and len(recent_bot_responses) > 0:
            original_resp = resp
            resp_lower = resp.lower().strip()

            # Check for exact duplicates first
            for prev_resp in recent_bot_responses[-5:]:
                prev_lower = prev_resp.lower().strip()

                # Exact match
                if resp_lower == prev_lower:
                    print(f"[ANTI-REPETITION] EXACT DUPLICATE DETECTED!")
                    print(f"[ANTI-REPETITION] Current: '{resp[:100]}...'")
                    print(f"[ANTI-REPETITION] Previous: '{prev_resp[:100]}...'")
                    print(
                        f"[ANTI-REPETITION] MODEL IS STUCK IN LOOP - This response will be rejected"
                    )
                    # Replace with error message that shows the model failed
                    resp = "I notice I'm repeating myself. Let me provide a different response."
                    break

                # Check for high similarity (>70% of words match)
                resp_words = set(resp_lower.split())
                prev_words = set(prev_lower.split())
                if len(resp_words) > 3 and len(prev_words) > 3:
                    overlap = len(resp_words.intersection(prev_words))
                    similarity = overlap / max(len(resp_words), len(prev_words))

                    if similarity > 0.7:
                        print(
                            f"[ANTI-REPETITION] HIGH SIMILARITY DETECTED ({similarity:.1%})"
                        )
                        print(f"[ANTI-REPETITION] Current: '{resp[:100]}...'")
                        print(f"[ANTI-REPETITION] Previous: '{prev_resp[:100]}...'")
                        print(
                            f"[ANTI-REPETITION] Replacing with loop-breaking response"
                        )
                        resp = "I seem to be giving similar responses. Let me try a different approach."
                        break

            # NEW: Check for phrase-level repetition (n-grams)
            # Extract phrases from current response, but be less aggressive
            if resp == original_resp:  # Only check if not already replaced

                def extract_ngrams(text, n):
                    """Extract n-word phrases from text"""
                    words = text.split()
                    return [
                        " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
                    ]

                def is_common_phrase(phrase):
                    """Filter out common professional phrases that are okay to repeat"""
                    common = [
                        "i can help",
                        "let me know",
                        "feel free to",
                        "happy to help",
                        "sure thing",
                        "sounds good",
                        "i'd be happy",
                        "would you like",
                        "let me explain",
                        "here's how",
                    ]
                    # Check if phrase contains any common patterns
                    for common_phrase in common:
                        if common_phrase in phrase:
                            return True
                    return False

                # Only check 6+ word phrases (more specific) to reduce false positives
                current_phrases_6 = set(extract_ngrams(resp_lower, 6))

                # Also track opening lines (first 5 words) as these are most noticeable
                resp_words = resp_lower.split()
                current_opening = (
                    " ".join(resp_words[:5]) if len(resp_words) >= 5 else None
                )

                for prev_resp in recent_bot_responses[
                    -3:
                ]:  # Only check last 3 responses
                    prev_lower = prev_resp.lower().strip()
                    prev_phrases_6 = set(extract_ngrams(prev_lower, 6))

                    # Check for repeated 6-word phrases (very specific, less false positives)
                    repeated_6 = current_phrases_6.intersection(prev_phrases_6)
                    # Filter out common phrases
                    repeated_6 = {p for p in repeated_6 if not is_common_phrase(p)}

                    if repeated_6:
                        print(f"[ANTI-REPETITION] ⚠️ REPEATED 6-WORD PHRASE DETECTED!")
                        for phrase in list(repeated_6)[:2]:  # Show first 2
                            print(f"[ANTI-REPETITION] Phrase: '{phrase}'")
                        print(f"[ANTI-REPETITION] WARNING: Consider regenerating")
                        # Don't immediately reject - just flag it
                        # resp = "hold up... i think i'm repeating myself. lemme switch it up bruh"
                        # For now, just log and allow it
                        break

                    # Check if opening line is exactly the same (this is always bad)
                    prev_words = prev_lower.split()
                    prev_opening = (
                        " ".join(prev_words[:5]) if len(prev_words) >= 5 else None
                    )

                    if (
                        current_opening
                        and prev_opening
                        and current_opening == prev_opening
                    ):
                        print(f"[ANTI-REPETITION] REPEATED OPENING LINE!")
                        print(f"[ANTI-REPETITION] Opening: '{current_opening}'")
                        print(f"[ANTI-REPETITION] This is noticeable - rejecting")
                        resp = "Let me rephrase my response to be more helpful."
                        break

            if resp != original_resp:
                print(f"[ANTI-REPETITION] Response replaced to break loop")

        # For DMs, add response to chatlogDM
        if is_dm:
            resp_dict = {"role": "assistant", "content": resp}
            chatlog.append(resp_dict)

        # Extract and store new facts about the user (only in DMs for now)
        # For group channels, facts are extracted in batch during summarization
        if is_dm:
            try:
                # CRITICAL: Never extract facts when username is the bot itself
                # This prevents the bot from storing facts about its own statements
                bot_names = [modelName, modelName.lower(), "assistant", "Assistant"]
                if username in bot_names:
                    print(
                        f"[MEMORY] Skipping fact extraction - username '{username}' is the bot itself"
                    )
                else:
                    print(
                        f"[STANDARD CHAT] Starting fact extraction from conversation..."
                    )
                    new_facts = await extract_user_facts(
                        username, cut_string, resp, user_activity_info
                    )
                    for fact in new_facts:
                        if user_memory.add_user_fact(username, fact):
                            print(f"[MEMORY] Stored new fact: {username} - {fact}")

                    # Extract event memories from recent conversation
                    # Convert chatlog to message format for event extraction
                    print(
                        f"[STANDARD CHAT] Starting event extraction from conversation..."
                    )
                    recent_messages = []
                    current_time = datetime.datetime.now()
                    for i, msg in enumerate(chatlog[-10:]):  # Last 10 messages
                        msg_time = (
                            current_time - datetime.timedelta(minutes=10 - i)
                        ).strftime("%Y-%m-%d %H:%M")
                        msg_username = username if msg["role"] == "user" else modelName
                        recent_messages.append(
                            {
                                "username": msg_username,
                                "content": msg["content"],
                                "timestamp": msg_time,
                                "is_bot": msg["role"] == "assistant",
                            }
                        )

                    new_events = await extract_event_memories(username, recent_messages)
                    for event in new_events:
                        if user_memory.add_event_memory(username, event):
                            print(
                                f"[EVENT MEMORY] Stored new event: {username} - {event}"
                            )
            except Exception as e:
                print(f"[ERROR] Error processing memories: {e}")

    # Send response to user (use reply to create threaded conversation)
    print(f"[RESPONSE] Sending {len(resp)} chars to user")
    await send_long_message(message, resp)

    # Retrieve and send relevant images
    try:
        import os

        images_sent = 0

        # For "show-images" intent: Always send doc images + web images
        # For other intents: Only send doc images if relevant
        if msg_intent == "show-images":
            # Step 1: Send documentation images with citations
            relevant_images = await asyncio.to_thread(
                retrieve_relevant_images, cut_string, 3
            )

            if relevant_images:
                print(f"[IMAGES] Found {len(relevant_images)} documentation images")
                for img_data in relevant_images:
                    image_path = img_data["image_path"]
                    source = img_data["source"]
                    page = img_data["page"]
                    score = img_data.get("relevance_score", 0)

                    print(
                        f"[IMAGES] Sending doc image: {image_path} (score: {score:.2f})"
                    )

                    try:
                        if os.path.exists(image_path):
                            file = discord.File(image_path)
                            caption = f"*{source}, Page {page}*"
                            await message.channel.send(caption, file=file)
                            images_sent += 1
                        else:
                            print(f"[IMAGES] Image file not found: {image_path}")
                    except Exception as img_send_error:
                        print(f"[IMAGES] Error sending image: {img_send_error}")
            else:
                print(f"[IMAGES] No documentation images found")

            # Step 2: Always search web for additional images
            print(f"[WEB IMAGES] Searching web for images...")
            web_images = await search_web_images(cut_string, max_images=2)

            for img_path in web_images:
                try:
                    if os.path.exists(img_path):
                        file = discord.File(img_path)
                        caption = "*Web search result*"
                        await message.channel.send(caption, file=file)
                        images_sent += 1
                except Exception as img_send_error:
                    print(f"[WEB IMAGES] Error sending web image: {img_send_error}")

        else:
            # For standard/internet intents: Only send doc images if relevant
            relevant_images = await asyncio.to_thread(
                retrieve_relevant_images, cut_string, 2
            )

            if relevant_images:
                print(
                    f"[IMAGES] Found {len(relevant_images)} relevant images for query"
                )
                for img_data in relevant_images:
                    image_path = img_data["image_path"]
                    source = img_data["source"]
                    page = img_data["page"]
                    score = img_data.get("relevance_score", 0)

                    print(f"[IMAGES] Sending image: {image_path} (score: {score:.2f})")

                    try:
                        if os.path.exists(image_path):
                            file = discord.File(image_path)
                            caption = f"*{source}, Page {page}*"
                            await message.channel.send(caption, file=file)
                            images_sent += 1
                        else:
                            print(f"[IMAGES] Image file not found: {image_path}")
                    except Exception as img_send_error:
                        print(f"[IMAGES] Error sending image: {img_send_error}")

        if images_sent == 0:
            print(f"[IMAGES] No images sent")
        else:
            print(f"[IMAGES] Sent {images_sent} images total")

    except Exception as e:
        print(f"[IMAGES] Error retrieving images: {e}")

    # Add bot response to channel history (group channels only - it's the source of truth)
    if not is_dm:
        print(f"[CHANNEL HISTORY] Adding bot response to channel_history")
        channel_history.add_message(
            channel_id=channel_id,
            username=modelName,
            content=resp,
            metadata={
                "username": modelName,
                "content": resp,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "is_bot": True,
                "author_id": client.user.id,
            },
        )

    print(f"[MESSAGE PROCESSING] Complete\n{'=' * 80}\n")


# ============================================================================
# SLASH COMMANDS - User Information
# ============================================================================
@client.tree.command(name="userinfo", description="prints the info of a user")
async def userinfo(interaction: discord.Interaction, member: discord.Member = None):
    """Display detailed information about a Discord user"""
    if member == None:
        member = interaction.user

    roles = [role for role in member.roles]

    # Create embed with user information
    embed = discord.Embed(
        title="user info",
        description=f"here is the info i found for the user {member.mention}",
        color=discord.Color.green(),
        timestamp=datetime.datetime.now(),
    )
    embed.set_thumbnail(url=member.avatar)
    embed.add_field(name="id", value=member.id)
    embed.add_field(name="name", value=f"{member.name}#{member.discriminator}")
    embed.add_field(name="nickname", value=member.display_name)
    embed.add_field(name="status", value=member.status)
    embed.add_field(
        name="created",
        value=member.created_at.strftime("%a, %B, %#d, %Y, %I:%M %p ").lower(),
    )
    embed.add_field(
        name="joined",
        value=member.joined_at.strftime("%a, %B, %#d, %Y, %I:%M %p ").lower(),
    )
    embed.add_field(
        name=f"roles ({len(roles)})", value=" ".join([role.mention for role in roles])
    )

    await interaction.response.send_message(embed=embed)


@client.tree.command(
    name="serverinfo", description="prints the info of the current server"
)
async def serverinfo(interaction: discord.Interaction):
    """Display detailed information about the current Discord server"""
    embed = discord.Embed(
        title="server info",
        description=f"here is the info i found for the server {interaction.guild.name}",
        color=discord.Color.blue(),
        timestamp=datetime.datetime.now(),
    )
    embed.set_thumbnail(url=interaction.guild.icon)
    embed.add_field(name="members", value=interaction.guild.member_count)
    embed.add_field(name="text channels", value=len(interaction.guild.text_channels))
    embed.add_field(name="voice channels", value=len(interaction.guild.voice_channels))
    embed.add_field(name="owner", value=interaction.guild.owner.mention)
    embed.add_field(name="description", value=interaction.guild.description)
    embed.add_field(
        name="created",
        value=interaction.guild.created_at.strftime(
            "%a, %B, %#d, %Y, %I:%M %p "
        ).lower(),
    )

    await interaction.response.send_message(embed=embed)


# ============================================================================
# SLASH COMMANDS - Chat History Management
# ============================================================================
@client.tree.command(name="clear", description="clear my chat history")
async def clear(interaction):
    """Clear all chat history (both DM and group)"""
    # Clear channel history if in a group channel
    if interaction.channel and not isinstance(interaction.channel, discord.DMChannel):
        channel_id = interaction.channel.id
        if channel_id in channel_history.channels:
            channel_history.clear_messages(channel_id)
            await interaction.response.send_message(
                "`chat history cleared for this channel, starting fresh`"
            )
        else:
            await interaction.response.send_message(
                "`no chat history found for this channel`"
            )
    else:
        # In DM - clear chatlogDM
        chatlogDM.clear()
        await interaction.response.send_message("`DM chat log cleared, starting fresh`")


@client.tree.command(name="save", description="save my chat history")
async def save(interaction):
    """Save DM chat history to JSON file on server"""
    export_to_json(chatlogDM, f"{myModel}-log.json")
    await interaction.response.send_message("`my logs have been saved to the server`")


@client.tree.command(name="load", description="load my last saved chat history")
async def load(interaction):
    """Load previously saved DM chat history from JSON file"""
    global chatlogDM
    loaded_data = import_from_json(f"{myModel}-log.json")
    chatlogDM = loaded_data if loaded_data is not None else []
    await interaction.response.send_message(
        "`my logs have been loaded from the server`"
    )


@client.tree.command(
    name="remember", description="show what the bot remembers about you"
)
async def remember(interaction):
    """Display all facts and events the bot has learned about the user"""
    username = interaction.user.display_name
    user_facts_with_meta = user_memory.get_user_facts_with_timestamps(username)
    user_events = user_memory.get_user_events(
        username, max_events=10
    )  # Get up to 10 recent events

    has_content = user_facts_with_meta or user_events

    if has_content:
        response_text = f"**Here's what I remember about you, {username}:**\n\n"

        # Add facts section organized by category
        if user_facts_with_meta:
            # Group facts by category
            facts_by_category = {}
            for fact_data in user_facts_with_meta:
                category = fact_data.get("category", "general")
                if category not in facts_by_category:
                    facts_by_category[category] = []
                facts_by_category[category].append(fact_data)

            # Display facts by category
            category_names = {
                "professional": "Professional",
                "personality": "Personality",
                "preferences": "Preferences",
                "technical": "Technical",
                "history": "History",
                "general": "General",
            }

            response_text += "**Facts:**\n"
            for category in [
                "professional",
                "personality",
                "preferences",
                "technical",
                "history",
                "general",
            ]:
                if category in facts_by_category:
                    category_display = category_names.get(
                        category, category.capitalize()
                    )
                    response_text += f"\n{category_display}:\n"
                    for fact_data in facts_by_category[category]:
                        text = fact_data.get("text", "")
                        response_text += f"  • {text}\n"
            response_text += "\n"

        # Add events section
        if user_events:
            response_text += "**Past Events:**\n"
            for i, event in enumerate(user_events, 1):
                event_desc = event.get("description", "")
                timestamp = event.get("timestamp", "")
                participants = event.get("participants", [])
                participants_str = (
                    ", ".join(participants) if participants else "unknown"
                )
                response_text += f"{i}. [{timestamp}] {event_desc}\n"
                response_text += f"   Participants: {participants_str}\n"

        await interaction.response.send_message(response_text)
    else:
        await interaction.response.send_message(
            f"I don't have any stored memories about you yet, {username}."
        )


@client.tree.command(
    name="forget", description="clear all memories the bot has about you"
)
async def forget(interaction):
    """Clear all stored memories for the user"""
    username = interaction.user.display_name

    if user_memory.clear_user_memories(username):
        await interaction.response.send_message(
            f"I've forgotten everything about you, {username}."
        )
    else:
        await interaction.response.send_message(
            f"I don't have any memories about you to forget, {username}."
        )


@client.tree.command(name="tasks", description="show your recent task history")
async def tasks(interaction):
    """Display recent tasks the user has requested"""
    username = interaction.user.display_name
    user_tasks = user_memory.get_user_tasks(username)

    if user_tasks:
        tasks_text = f"**Here are your recent tasks, {username}:**\n\n"
        for i, task in enumerate(user_tasks[-5:], 1):  # Show last 5 tasks
            task_type = task.get("type", "unknown")
            request = task.get("request", "")
            outcome = (
                task.get("outcome", "")[:100] + "..."
                if len(task.get("outcome", "")) > 100
                else task.get("outcome", "")
            )
            timestamp = task.get("timestamp", "")
            tasks_text += f"{i}. **[{task_type}]** {timestamp}\n"
            tasks_text += f"   Request: {request}\n"
            tasks_text += f"   Result: {outcome}\n\n"
        await interaction.response.send_message(tasks_text)
    else:
        await interaction.response.send_message(
            f"You haven't asked me to do any specific tasks yet, {username}."
        )


@client.tree.command(
    name="cleanup_memories",
    description="remove duplicate and similar facts using AI (admin only)",
)
async def cleanup_memories(interaction: discord.Interaction):
    """Deduplicate stored memories using semantic similarity"""
    # Check if user has admin permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ Only server administrators can use this command.", ephemeral=True
        )
        return

    await interaction.response.defer()  # This might take a moment

    try:
        results = user_memory.deduplicate_all_facts()

        if results:
            response = "**🧹 Memory Cleanup Complete**\n\n"
            response += "Removed duplicate/similar facts using semantic AI:\n\n"
            for username, count in results.items():
                response += f"• **{username}**: {count} duplicate(s) removed\n"
            await interaction.followup.send(response)
        else:
            await interaction.followup.send(
                "✨ No duplicates found! Memory is already clean."
            )

    except Exception as e:
        await interaction.followup.send(f"❌ Error during cleanup: {str(e)}")


@client.tree.command(
    name="reload_docs",
    description="reload training documentation from disk (admin only)",
)
async def reload_docs(interaction: discord.Interaction):
    """Reload training documentation into the vector store"""
    # Check if user has admin permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ Only server administrators can use this command.", ephemeral=True
        )
        return

    await interaction.response.defer()  # This might take a moment

    try:
        # Clear existing vector store and reload
        global _vector_store
        _vector_store = None  # Reset to force reload

        doc_count = load_training_documents()

        if doc_count > 0:
            await interaction.followup.send(
                f"📚 **Documentation Reloaded**\n\nSuccessfully loaded {doc_count} document chunks from `{DOCS_DIRECTORY}`"
            )
        else:
            await interaction.followup.send(
                f"⚠️ No documents found in `{DOCS_DIRECTORY}`\n\nAdd `.txt` files to enable documentation retrieval."
            )

    except Exception as e:
        await interaction.followup.send(f"❌ Error reloading documentation: {str(e)}")


@client.tree.command(
    name="doc_status",
    description="check the status of the documentation system",
)
async def doc_status(interaction: discord.Interaction):
    """Check how many documents are loaded in the vector store"""
    try:
        vector_store = get_vector_store()
        doc_count = vector_store._collection.count()

        if doc_count > 0:
            await interaction.response.send_message(
                f"📚 **Documentation Status**\n\n"
                f"• Document chunks loaded: **{doc_count}**\n"
                f"• Source directory: `{DOCS_DIRECTORY}`\n"
                f"• Vector store: `{VECTOR_STORE_PATH}`"
            )
        else:
            await interaction.response.send_message(
                f"⚠️ **No Documents Loaded**\n\n"
                f"Add `.txt` files to `{DOCS_DIRECTORY}` and run `/reload_docs`"
            )

    except Exception as e:
        await interaction.response.send_message(
            f"❌ Error checking documentation: {str(e)}"
        )


# ============================================================================
# BOT EXECUTION
# ============================================================================
client.run(botToken)
