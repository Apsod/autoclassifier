from dataclasses import dataclass
from collections.abc import Sequence
import logging
import math

from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import pairwise_distances_argmin_min

from autoclassifier.util import Labeled
import numpy as np

logger = logging.getLogger(__name__)

###########
# LEARNER #
###########

class Learner(object):
    def __init__(self, data, external=None, seen=None, model_kwargs={'C': 1e1, 'max_iter': 10_000}):
        self.data = data
        self.svc = LinearSVC(**model_kwargs)
        self.external = external
        self.train = seen if seen is not None else {}
        self.fit()

    @property
    def labeled(self):
        return [Labeled(i,l) for i, l in self.train.items() if l is not None]

    @property
    def ixs(self):
        return [l.index for l in self.labeled]

    @property
    def ls(self):
        return [l.label for l in self.labeled]

    @property
    def N(self):
        N_i = len(self.train)
        if self.external:
            N_e = len(self.external[1])
            return N_i + N_e
        return N_i

    def get_texts(self, indices=slice(None)):
        return self.data[indices, 'text'].to_list()

    def get_embeddings(self, indices=slice(None)):
        return self.data[indices, 'embedding'].to_numpy()
    
    def get_datapoints(self):
        xs_i = self.get_embeddings(self.ixs)
        ys_i = np.array(self.ls)

        if self.external:
            xs_e, ys_e = self.external
            return np.concatenate((xs_e, xs_i)), np.concatenate((ys_e, ys_i))
        else:
            return xs_i, ys_i

    @property
    def xs(self):
        return self.get_datapoints()[0]
    
    @property
    def ys(self):
        return self.get_datapoints()[1]

    def fit(self):
        self.svc.fit(*self.get_datapoints())

    def measure(self, embeddings=None):
        if embeddings is None:
            embeddings = self.get_embeddings()
        return self.svc.decision_function(embeddings)

    def distance_score(self, embeddings):
        k = math.ceil(math.sqrt(self.N))
        km = KMeans(n_clusters=k)
        km.fit(self.xs)
        cid, ds = pairwise_distances_argmin_min(embeddings, km.cluster_centers_)
        return ds

    def uncertainty_score(self, embeddings):
        return np.abs(self.measure(embeddings))

    def combined_score(self, embeddings, joint_w=0.0, distance_w=0.0, uncertainty_w=1.0):
        """
        Combined uncertainty and distance based sampling. 
        distance_score: each point has a score between 0 and 1 based on the distance from the labeled dataset.
            Scores go from farthest (0) to closest (1).
        uncertainty_score gives each point a score between 0 and 1 based on the distance from the decision boundary.
            Scores go from closest (0) to farthest (1).

        joint_w: A weight applied to distance_score * uncertainty_score
        distance_w: A weight applied only to distance_score
        uncertainty_w: A weight applied only to uncertainty_score

        default: joint_w = 0.5, distance_w = 0, uncertainty_w = 1 => combined_score = distance_score * uncertainty_score/2 + uncertainty_score
        """
        return self.uncertainty_score(embeddings)
        #ds = QuantileTransformer().fit_transform(-self.distance_score(embeddings).reshape(-1, 1)).reshape(-1)
        #us = QuantileTransformer().fit_transform(self.uncertainty_score(embeddings).reshape(-1, 1)).reshape(-1)
        #return ds * us * joint_w + ds * distance_w + us * uncertainty_w

    def kmeans_select(self, k: int, n: None | int = None, exclude: None | Sequence[int] = None):
        if k == 0:
            return set()
        if n is None:
            n = math.ceil(math.sqrt(len(self.data) - len(self.train)))
            #n = k * k
        exclude = exclude if exclude else set()
        exclude = exclude | set(self.train)
        
        # Extract unseen indices
        ixs = np.arange(len(self.data))
        ixs = ixs[np.isin(ixs, list(exclude), assume_unique=True, invert=True)]
        
        # Get scores
        scores = self.combined_score(self.get_embeddings(ixs))

        # Get top scoring indices
        ixs = ixs[np.argsort(scores)[:n]]

        # Cluster top scorers
        x = self.get_embeddings(ixs)
        y = KMeans(n_clusters=k).fit_predict(x)
            
        # Get the top scoring candidate from each cluster.
        ret = set()
        for i in range(k):
            try:
                ret.add(ixs[y==i][0].item())
            except IndexError:
                logger.warning('Empty cluster in kmeans_select')
        return ret

    def score(self):
        return self.svc.score(*self.get_datapoints())

    def cross_val_score(self, *args, **kwargs):
        return np.mean(cross_val_score(self.svc, *self.get_datapoints(), *args, **kwargs)).item()

    def sample_unlabeled(self, k):
        # Extract unseen indices
        ixs = np.arange(len(self.data))
        ixs = ixs[np.isin(ixs, list(self.train), assume_unique=True, invert=True)]
        return [ix.item() for ix in np.random.choice(ixs, size=k, replace=False)]

    def get_candidates(self, k, exclude=None):
        return list(self.kmeans_select(k, exclude=exclude))

    def update(self, labeled: Sequence[Labeled]):
        """
        add the labeled datapoints to train.
        datapoints can be unlabeled (due to model refusal),
        in which case they are added to train, but ignored when
        extracting training datapoints.
        """
        for datum in labeled:
            self.train[datum.index] = datum.label
        self.fit()

    def stats(self, labeled: Sequence[labeled] = None):
        if labeled is None:
            ixs = self.ixs
            ls = self.ls
        else:
            ixs = [l.index for l in labeled]
            ls = [l.label for l in labeled]

        scores = self.measure(self.get_embeddings(ixs)).tolist()
        stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'loss': 0}

        total = 0

        for s, l in zip(scores, ls):
            total += 1
            g = 1 if l else -1
            stats['loss'] += max(1 - s * g, 0)
            match (s > 0, l):
                case True, 1:
                    stats['tp'] += 1
                case True, 0:
                    stats['fp'] += 1
                case False, 0:
                    stats['tn'] += 1
                case False, 1:
                    stats['fn'] += 1

        # hack to deal with total = 0 ...
        if total == 0:
            logging.warning('Stat calculation with no valid labeled datapoints')
            total = 1

        total = total if total > 0 else 1

        return {k: v / total for k, v in stats.items()}

    def save(self, save_dir):
        p = pathlib.Path(save_dir)
        p.parent.mkdir(exist_ok=True)
        with open(p / 'model.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=5)
