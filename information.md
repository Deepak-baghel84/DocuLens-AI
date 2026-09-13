Api key store as a json file in production 
Ya i have used lot of ai, making my project suitable for deployment.
Previously i have used gemini embedding model now i changed it with sentence transformer embedding model.
SentenceTransformer usually runs the model locally so we does not required any api key but You may need a Hugging Face token only for certain private or gated models
When i installed HuggingFace package it internally install some other libraries like torch,etc.. with a specific version but these are already installed with other version creates a version problem.

install all packages using python -m pip install .....

deepeval also provide custom dataset(goldens) for evaluation, cloud infrastructure for dashboard and evaluation result. have to pay for that.
But i done it locally created custom tests cases with actual answers and context for that purpose i used chatgpt.

Remember evaluation dataset called as goldens.

Evaluation happens llm as a judge so by default deepeval uses openai api for that so i need to create a class to do evaluation using chatgroq 

At this point DeepEval tries to initialize an LLM judge.

The default DeepEval metrics are LLM-as-a-judge metrics, and when you don't specify a judge model, DeepEval defaults to OpenAI.
DeepEval's metrics are LLM-as-a-judge metrics, and you can provide a custom model instead of OpenAI.created GroqEval custom llm as a judge where i consider temperature as 0 .

pdf contains text as images got failed during evaluation 

groq.RateLimitError      service tier `on_demand` on tokens per minute (TPM): Limit 8000
so this 8000 token include complete input_tokens(query+context+prompt+history) + output_token
all 6 metrices starts executing together with same model so each time it shows 429.







