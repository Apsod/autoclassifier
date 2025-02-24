# What is this?

This is a simple active learning loop for training an SVM on embedded documents, where the labeling of the document is outsourced to an LLM (Gemini, in the current implementation). 

In `script/run` there's a simple script for running the loop given an API-key and a parquet file with a `text` and `embedding` column.
By default, the script will seed the classifier with examples mocked by an LLM, in which case you also need to embed these documents. In the script this is done using `intfloat/multilingual-e5-small`, change the script to accomodate different embeddings.

## How does it work?

The active learning loop consists of three parts, an Oracle, a Learner, and a Runner which binds the Oracle and Learner together.

### Oracle

`src/autoclassifier/oracle.py`

The oracle is responsible for labeling documents. This is done using an LLM and a template which wrap a criteria list.
Given a template, the oracle can then either mock positive and negative documents, or label documents:

```
from autoclassifier.oracle import Template, Oracle
template = Template(criteria='* The document contains spurious content indicative of SEO keyword stuffing')
oracle = Oracle(template, api_key=YOUR_API_KEY)

reason, label = oracle.label('Hello and welcome to my blog about cats. What are cats? CASINO FREE SPINS WIN LOTS OF CASH. Cats are cute')
```

When labeling, the oracle gives a short reason motivating the answer, and a label (extracted from the response).

### Learner

`src/autoclassifier/learner.py`

The learner is responsible for keeping track of the underlying classifier and data, as well as selecting new candidates to label.
It uses a simple linear support vector classifier, applied to embedded documents, and selects new documents to label based on (clustered) uncertainty sampling.

### Runner

`script/run.py`

The runner (implemented in the example script), is responsible for tying the learner and oracle together. During a step, it first queries the learned for new candidates to label, sends them to the oracle, and then updates the learner with the labeled candidates. 
It also collects training statistics for logging.
