import math
import pathlib
import json
import pickle
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import polars as pl
import wandb
import numpy as np
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress
from rich.status import Status

from autoclassifier.oracle import Oracle, Template, RefusalError, ResponseError
from autoclassifier.learner import Learner
from autoclassifier.util import Labeled

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class Display(object):
    def __init__(self):
        self.status = Status('Init')
        self.pdict = {}
        self.progress = Progress()
        self.pbar_step = self.add_task('Step')
        self.pbar_label = self.add_task('Labeler')

    def set_status(self, status):
        self.status.update(status)

    def add_task(self, *args, **kwargs):
        return self.progress.add_task(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.progress.reset(*args, **kwargs)

    def advance(self, *args, **kwargs):
        self.progress.advance(*args, **kwargs)

    def reset_label(self, *args, **kwargs):
        self.reset(self.pbar_label, *args, **kwargs)

    def advance_label(self, *args, **kwargs):
        self.advance(self.pbar_label, *args, **kwargs)

    def advance_step(self, *args, **kwargs):
        self.advance(self.pbar_step, *args, **kwargs)

    def reset_step(self, *args, **kwargs):
        self.reset(self.pbar_step, *args, **kwargs)

    def __rich_console__(self, console, options):
        yield self.status
        yield self.progress.make_tasks_table(self.progress.tasks)

class Runner(object):
    def __init__(self, display, learner, oracle, max_workers=8, test_samples=10, samples_per_step=10):
        self.display = display
        self.learner = learner
        self.oracle = oracle
        self.max_workers = max_workers
        self.samples_per_step = samples_per_step
        self.test_samples = list(self.label_candidates(self.learner.sample_unlabeled(test_samples), which='test'))


    def get_candidates(self):
        self.display.set_status('Getting candidates')
        return self.learner.get_candidates(self.samples_per_step, exclude={i for i, t, r, l in self.test_samples})

    def label_candidates(self, candidates, which='train'):
        self.display.set_status(f'Labeling {which} candidates')
        idx2txt = {idx:txt for idx, txt in zip(candidates, self.learner.get_texts(candidates))}
        self.display.reset_label(total=len(idx2txt))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut2idx = {executor.submit(self.oracle.label, txt): idx for idx, txt in idx2txt.items()}
            for future in as_completed(fut2idx):
                self.display.advance_label()
                idx = fut2idx[future]
                try:
                    reason, label = future.result()
                    yield (idx, idx2txt[idx], reason, label)
                except (RefusalError, ResponseError) as e:
                    logger.warning(f'{e}')

    def update(self, labeled: Sequence[Labeled]):
        self.display.set_status('Updating model')
        self.learner.update(labeled)

    def step(self):
        cs = self.get_candidates() # Get candidates
        results = list(self.label_candidates(cs)) # Label candidates

        labeled = [Labeled(i, l) for i, t, r, l in results]
        step_stats = self.learner.stats(labeled)

        self.update(labeled) # Update model
        train_stats = self.learner.stats()

        test_stats = self.learner.stats([Labeled(i,l) for i, t, r, l in self.test_samples])

        stats = {}
        stats.update({f'train/{k}': v for k, v in train_stats.items()})
        stats.update({f'step/{k}': v for k, v in step_stats.items()})
        stats.update({f'test/{k}': v for k, v in test_stats.items()})
        stats['cross_val/score'] = self.learner.cross_val_score()
        stats['data/positive_ratio'] = (self.learner.measure() > 0).mean().item()
        self.display.advance_step()
        return stats, results

def embed(texts):
    """
    I've used intfloat multilingual, put other stuff here if you use other embeddings.
    """
    import torch
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel
    with torch.no_grad():
        def average_pool(last_hidden_states: Tensor,
                         attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        model = AutoModel.from_pretrained('intfloat/multilingual-e5-small').eval()
        batch = tokenizer(['query: {}'.format(text) for text in texts], max_length=512, padding=True, truncation=True, return_tensors='pt')
        emb = average_pool(
                model(**batch).last_hidden_state,
                batch['attention_mask']
                ).numpy()
    return emb

def mock_external(oracle, positive=None, negative=None):
    with ThreadPoolExecutor() as executor:
        if positive is None:
            positive_fut = executor.submit(oracle.mock, 'pos')
        if negative is None:
            negative_fut = executor.submit(oracle.mock, 'neg')
        if positive is None:
            positive = positive_fut.result()
        if negative is None:
            negative = negative_fut.result()
    logger.info('Successfully mocked seed data')
    logger.info(f'Positive:\n{positive}')
    logger.info(f'Negative:\n{negative}')
    xs = embed([positive, negative])
    ys = np.array([1, 0])
    return xs, ys

def run_with(api_key, save_dir, data, criteria, wandb_kwargs={}, positive=None, negative=None, samples_per_step=10, steps=40, test_samples=200):
    display = Display()
    display.reset_step(total=steps)
    logdir = pathlib.Path(save_dir)
    logdir.mkdir(exist_ok=True)
    with open(logdir / 'train.json', 'wt') as logf, Live(display):
        # Setting up
        oracle = Oracle(
                api_key=api_key,
                template = Template(criteria, language='Swedish'),
                )

        wandb_logger = wandb.init(
                **wandb_kwargs,
                config={
                    'criteria': criteria,
                    'steps': steps
                },
        )


        display.set_status('(Maybe?) Mocking data')
        external = mock_external(oracle, positive=positive, negative=negative)
        learner = Learner(data, external=external)
        runner = Runner(display, learner, oracle, samples_per_step=samples_per_step, test_samples=test_samples)
        with open(logdir / 'test.json', 'wt') as testf:
            for (i, t, r, l) in runner.test_samples:
                testf.write(json.dumps({'id': i, 'text': t, 'reason': r, 'label': l}))
                testf.write('\n')

        # Running
        for step in range(steps):
            stats, results = runner.step()
            display.set_status('Logging')
            for (i, t, r, l) in results:
                logf.write(json.dumps({'id': i, 'text': t, 'reason': r, 'label': l}))
                logf.write('\n')

            wandb_logger.log(
                    data = stats,
                    step = step)
        
        display.set_status('Saving model')
        with open(logdir / 'model.bin', 'wb') as mfile:
            pickle.dump(runner.learner.svc, mfile, protocol=5)

PROMPT = dict(
        qa = """\
                * The text contains at least one question, and an answer to that question.
                * The question should be clear and concise, and should require logical deduction or drawing on multiple pieces of knowledge.
                * The answer must be a clear, logical, and comprehesive explanation of how to arrive at the given answer.
                * The answer breaks the problem down into smaller, manageable steps.
                * The answer explains the reasoning behind each step.""",
        content = """\
                * The text contains at least two paragraphs of coherent, meaningful text.
                * If the text contain tabular or list form data, such data should be limited in scope and pertinent to the main content of the text.""",
        porn_etc = """\
                * The main purpose of the text is to promote a service.
                * The service being promoted is related to topics such as porn, prostitution, dating, gambling, or betting.""",
        seo = """\
                * The text contains spurious content indicative of SEO keyword stuffing.""",
        quality = """\
                * Clarity: The text is easy to understand; avoids ambiguity.
                * Conciseness: The text expresses ideas efficiently, avoiding unnecessary words.
                * Coherence: The text is logically organized and flows smoothly.
                * Relevance: The text stays focused on the topic and purpose.
                * Correctness: The text is free of grammatical errors, spelling mistakes, and punctuation issues.
                * Purposeful: The text achieves its intended goal (inform, persuade, entertain, etc.).""",
        quality_relaxed = """\
                * Clarity: The text is easy to understand; avoids ambiguity.
                * Coherence: The text is logically organized and flows smoothly.
                * Relevance: The text stays focused on the topic and purpose.
                * Purposeful: The text achieves its intended goal (inform, persuade, entertain, etc.).""",
        convo = """\
                * The text contains a conversation between at least two individuals.""",
        toxic = """\
                * The text is toxic or otherwise harmful.""",
        promotional = """\
                * The main purpose of the text is to promote a service or a product.""",
)

DATA_PATH = 'path/to/data.parquet'
WANDB_PROJECT = 'autoclassifier'

if __name__ == '__main__':
    df = pl.read_parquet()
    key = 'seo'

    run_with(
            api_key=os.getenv('GEMINI_KEY'),
            save_dir=key,
            data=df,
            criteria = PROMPT[key],
            wandb_kwargs = {'project': WANDB_PROJECT},
            steps=100,
            )

