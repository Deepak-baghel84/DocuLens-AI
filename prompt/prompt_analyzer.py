from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template("""
You are a precise document comparison assistant. Compare the content of two PDFs and return ONLY a valid JSON output.

Follow these strict rules:
- Compare the documents page by page.
- For each page, if there are differences, describe them under "Changes".
- If no changes exist, write "NO CHANGE".
- Do not add extra commentary or explanations.
- Your entire response must be a JSON list, nothing else.

Expected JSON Format:
[
  {{
    "Page": "<page_number or name>",
    "Changes": "<description of changes or 'NO CHANGE'>"
  }}
]

Documents to compare:
{combined_docs}
""")

# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Given the conversation history and the latest user question:

- Rewrite the question as a complete standalone question when necessary.
- Preserve the original meaning and intent.
- Resolve references such as 'it', 'they', 'this', 'that', or follow-up questions using the conversation history.
- Do not answer the question.
- Do not add new information.
- If the question is already standalone, return it unchanged.
- Return only the rewritten question.
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{user_input}")
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent document assistant.

Your primary task is to answer the user's question using the provided document context.

Instructions:

1. Carefully analyze all retrieved context before answering.
2. Use only information present in the context.
3. Combine information from multiple context sections when necessary.
4. Provide clear, complete, and well-structured answers.
5. If the context contains only partial information, answer using the available information and clearly state what is missing.
6. Do not invent facts, assumptions, or information that is not explicitly supported by the context.
7. If the answer cannot be found in the context, respond:
   "I could not find enough information in the provided documents to answer this question."
8. When appropriate, use bullet points, numbered lists, or tables for clarity.
9. Preserve important technical details, names, dates, figures, and definitions exactly as they appear in the documents.
10. If the user asks for a summary, provide a concise summary of the relevant information.
11. If the user asks for an explanation, provide a detailed explanation based solely on the context.

Retrieved Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{user_input}")
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}
