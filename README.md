# What is this?

This is a simple active learning loop for training an SVM on embedded documents, where the labeling of the document is outsourced to an LLM (Gemini, in the current implementation). 

In `script/run` there's a simple script for running the loop given an API-key, and a parquet file with a `text` column and an `embedding` column.
By default, the script will seed the classifier with examples mocked by an LLM, in which case you also need to embed these documents. In the script this is done using `intfloat/multilingual-e5-small`, change the script to accomodate different embeddings.
