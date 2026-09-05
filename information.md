Api key store as a json file in production 
Ya i have used lot of ai, making my project suitable for deployment.
Previously i have used gemini embedding model now i changed it with sentence transformer embedding model.
SentenceTransformer usually runs the model locally so we does not required any api key but You may need a Hugging Face token only for certain private or gated models
When i installed HuggingFace package it internally install some other libraries like torch,etc.. with a specific version but these are already installed with other version creates a version problem.
