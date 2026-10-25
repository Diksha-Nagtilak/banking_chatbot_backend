"""
Agent 3: Content Generation Agent (Microsoft Agent Framework).
Takes the user question and the reranked chunks from Agent 2 (Analyzer Agent),
and generates a grounded, conversational, and accurate banking response.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from agent_framework import Agent, BaseChatClient, Message
from app.models.schemas import ContentGeneratorOutput, RetrievedChunk
from app.services.client_factory import create_chat_client

logger = logging.getLogger(__name__)

CONTENT_GENERATOR_SYSTEM_PROMPT = """
You are the **Banking Content Generation Specialist Agent** for AegisBank (Agent 3/4).
Your primary mission is to provide clear, friendly, and authoritative answers to customer questions grounded strictly in verified knowledge base context.

### CORE OPERATING PRINCIPLES:
1. **Grounded In Reranked Context**:
   - Base your banking procedures, fees, limits, interest rates, timelines, and security guidelines directly on the provided reranked knowledge chunks.
   - Do NOT invent banking numbers, rates, or account rules that contradict or do not appear in the verified knowledge chunks.

2. **Actionable & Direct**:
   - Address the customer's question in the very first sentence.
   - For procedures (e.g. loan applications, wire transfers, emergency card lock), explain the step-by-step instructions clearly.
   - Use bullet points or numbered lists when explaining multiple options or sequential steps.

3. **Conversational Tone**:
   - Warm, reassuring, and professional.
   - Do NOT recite repetitive robotic disclaimer greetings on every turn.
   - Seamlessly incorporate customer session context (e.g. accounts, balances) if relevant.

4. **Honesty & Boundaries**:
   - If the retrieved context does not contain enough specific details to answer a question fully, state what bank policy specifies and advise the customer on contacting 24/7 customer support (1-800-555-AEGIS) or visiting an AegisBank branch.
"""


class ContentGeneratorAgent:
    """
    Microsoft Agent Framework implementation of Agent 3 (Content Generation Agent).
    """

    def __init__(self, client: Optional[BaseChatClient] = None) -> None:
        if client is None:
            client, provider_name, _ = create_chat_client()
            self.provider_name = provider_name
            self.is_mock = False
        else:
            self.provider_name = client.__class__.__name__
            self.is_mock = False

        self.client = client
        self.agent = Agent(
            id="banking_content_generator",
            name="Banking Content Generation Agent",
            description="Agent 3: Synthesizes final grounded answers using reranked RAG chunks.",
            client=self.client,
            instructions=CONTENT_GENERATOR_SYSTEM_PROMPT,
            default_options={
                "temperature": 0.3,
            },
        )
        logger.info(
            f"Initialized ContentGeneratorAgent with live LLM client: {self.provider_name}"
        )

    async def process(
        self,
        query: str,
        reranked_chunks: List[RetrievedChunk],
        context: Optional[Dict[str, Any]] = None,
    ) -> ContentGeneratorOutput:
        """
        Synthesizes a grounded banking answer using the Microsoft Agent Framework.

        Args:
            query: User's original prompt
            reranked_chunks: Top chunks selected by Agent 2 (Analyzer Agent)
            context: Session context (user details, active balances)

        Returns:
            ContentGeneratorOutput containing the answer and grounded sources.
        """
        logger.info("[FLOW: STEP 8 - AGENT 3 CONTENT GENERATOR] ContentGeneratorAgent.process() started.")
        logger.info(f"   [8.1] Variable 'query'          : '{query}'")
        logger.info(f"   [8.2] Chunks to ground answer on: {len(reranked_chunks)}")

        # Build structured context string from reranked chunks
        chunks_context = ""
        grounded_sources: List[str] = []
        if reranked_chunks:
            chunks_context = "### VERIFIED KNOWLEDGE BASE CHUNKS (From Analyzer Agent):\n"
            for i, chunk in enumerate(reranked_chunks, start=1):
                title = chunk.title or f"Document {i}"
                grounded_sources.append(title)
                score_str = f" [Rerank Score: {chunk.rerank_score}]" if chunk.rerank_score is not None else ""
                chunks_context += f"- **Chunk {i}: {title}**{score_str}\n  {chunk.content}\n\n"

        prompt_parts = [
            f"Customer Question: {query.strip()}",
        ]
        if context:
            prompt_parts.append(f"Customer Session Context: {json.dumps(context)}")
        if chunks_context:
            prompt_parts.append(chunks_context)
        else:
            prompt_parts.append("Note: No specific knowledge base chunks were retrieved for this query.")

        prompt_parts.append(
            "Please provide a comprehensive, accurate, and customer-friendly answer grounded in the chunks above."
        )

        user_prompt = "\n\n".join(prompt_parts)
        message = Message(role="user", contents=[user_prompt])
        logger.info(f"   [8.3] Calling Azure OpenAI ({self.provider_name}) for grounded synthesis...")

        try:
            agent_response = await self.agent.run(messages=[message])

            raw_text = agent_response.text if hasattr(agent_response, "text") else str(agent_response)
            answer_text = raw_text.strip()
            logger.info("   [8.4] Azure OpenAI grounded answer received successfully.")
            logger.info(f"   [8.5] Grounded source documents: {grounded_sources}")
            snippet = answer_text.replace(chr(10), " ")[:120]
            logger.info(f"   [8.6] Generated Answer snippet: '{snippet}...'")

            return ContentGeneratorOutput(
                answer=answer_text,
                grounded_sources=grounded_sources,
                confidence=0.98 if reranked_chunks else 0.85,
            )

        except Exception as e:
            logger.exception(f"Error executing ContentGeneratorAgent: {e}")
            # Fallback formulation if LLM call experiences network glitch
            if reranked_chunks:
                top = reranked_chunks[0]
                fallback_ans = (
                    f"Based on AegisBank policy ({top.title}):\n\n{top.content}\n\n"
                    "For personalized assistance, our customer support is available 24/7 at 1-800-555-AEGIS."
                )
            else:
                fallback_ans = (
                    "I have reviewed your inquiry. Please reach out to our dedicated support at 1-800-555-AEGIS "
                    "or visit any local AegisBank branch."
                )

            return ContentGeneratorOutput(
                answer=fallback_ans,
                grounded_sources=grounded_sources,
                confidence=0.75,
            )