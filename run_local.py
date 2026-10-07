"""
Interactive CLI for testing the Banking Request Handler Agent locally.
Run with:
    python run_local.py
"""
import asyncio
import sys
from app.agents.request_handler import RequestHandlerAgent


async def main() -> None:
    print("=" * 70)
    print("  BANKING CHATBOT - REQUEST HANDLER AGENT (MICROSOFT AGENT FRAMEWORK)")
    print("=" * 70)

    agent = RequestHandlerAgent()
    print(f"Provider: {agent.provider_name} (Mock Mode: {agent.is_mock})")
    print("Agent is ready. Type your query below (or 'exit' / 'quit' to quit).\n")

    sample_queries = [
        "What is the balance in my checking account?",
        "Please transfer $250 to Sarah for dinner.",
        "Help! I lost my debit card ending in 4321, please block it!",
        "Can you share the recipe for chocolate cake?",
        "What are your current mortgage and 30-year home loan interest rates?",
        "How do I write a Python loop?",
    ]

    print("Suggested test queries:")
    for idx, q in enumerate(sample_queries, 1):
        print(f"  {idx}. {q}")
    print("-" * 70)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nEvaluating argument: '{query}'")
        output = await agent.process(query)
        print("\n--- AGENT RESULT ---")
        print(f"Banking Related    : {output.is_banking_related}")
        print(f"Category           : {output.category.value}")
        print(f"Detected Intent    : {output.detected_intent}")
        print(f"Recommended Agent  : {output.recommended_agent.value if output.recommended_agent else 'None'}")
        print(f"Enriched Query     : {output.enriched_query}")
        print(f"Entities Extracted : {output.entities.model_dump(exclude_none=True)}")
        print(f"Missing Info       : {output.missing_information}")
        print(f"Response Message   : {output.response_message}")
        print("-" * 70)
        return

    while True:
        try:
            user_input = input("\nEnter query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting. Goodbye!")
                break

            print("\nProcessing with Microsoft Agent Framework...")
            output = await agent.process(user_input)

            print("\n" + "=" * 50)
            print(f"Is Banking Related : {output.is_banking_related} (Confidence: {output.confidence:.2f})")
            print(f"Category           : {output.category.value}")
            print(f"Detected Intent    : {output.detected_intent}")
            print(f"Recommended Agent  : {output.recommended_agent.value if output.recommended_agent else 'None (Refusal)'}")
            if output.enriched_query:
                print(f"Enriched Query     : {output.enriched_query}")
            if output.entities.model_dump(exclude_none=True):
                print(f"Entities           : {output.entities.model_dump(exclude_none=True)}")
            if output.missing_information:
                print(f"Missing Information: {output.missing_information}")
            print(f"Agent Response     : {output.response_message}")
            print("=" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
