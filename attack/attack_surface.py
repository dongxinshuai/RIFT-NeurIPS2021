"""Defines an attack surface."""
import collections
import json
import sys

OPTS = None

DEFAULT_MAX_LOG_P_DIFF = -5.0  # Maximum difference in log p for swaps.

class AttackSurface(object):
  def get_swaps(self, words):
    """Return valid substitutions for each position in input |words|."""
    raise NotImplementedError

class WordSubstitutionAttackSurface(AttackSurface):
  def __init__(self, neighbors, lm_scores):
    self.neighbors = neighbors
    self.lm_scores = lm_scores

  @classmethod
  def from_files(cls, neighbors_file, lm_file):
    with open(neighbors_file) as f:
      neighbors = json.load(f)
    with open(lm_file, encoding="utf-8") as f:
      lm_scores = {}
      cur_sent = None
      for line in f:
        toks = line.strip().split('\t')
        if len(toks) == 2:
          cur_sent = toks[1].lower()
          lm_scores[cur_sent] = collections.defaultdict(dict)
        else:
          word_idx, word, score = int(toks[1]), toks[2], float(toks[3])
          lm_scores[cur_sent][word_idx][word] = score
    return cls(neighbors, lm_scores)


  def get_swaps(self, words):
    swaps = []
    for i in range(len(words)):
      word = words[i]
      if word in self.neighbors: 
        swaps.append(self.neighbors[word])
      else:
        swaps.append([])
    return swaps

  def check_in(self, words):
    words = [word.lower() for word in words]
    s = ' '.join(words)
    return s in self.lm_scores

class LMConstrainedAttackSurface(AttackSurface):
  """WordSubstitutionAttackSurface with language model constraint."""
  def __init__(self, neighbors, lm_scores, min_log_p_diff=DEFAULT_MAX_LOG_P_DIFF):
    self.neighbors = neighbors
    self.lm_scores = lm_scores
    self.min_log_p_diff = min_log_p_diff

  @classmethod
  def from_files(cls, neighbors_file, lm_file):
    with open(neighbors_file) as f:
      neighbors = json.load(f)
    with open(lm_file, encoding="utf-8") as f:
    #with open(lm_file) as f:
      lm_scores = {}
      cur_sent = None
      for line in f:
        toks = line.strip().split('\t')
        if len(toks) == 2:
          cur_sent = toks[1].lower()
          lm_scores[cur_sent] = collections.defaultdict(dict)
        else:
          word_idx, word, score = int(toks[1]), toks[2], float(toks[3])
          lm_scores[cur_sent][word_idx][word] = score
    return cls(neighbors, lm_scores)

  def get_swaps(self, words):
    swaps = []
    words = [word.lower() for word in words]
    s = ' '.join(words)
    if s not in self.lm_scores:
      raise KeyError('Unrecognized sentence "%s"' % s)
    for i in range(len(words)):
      if i in self.lm_scores[s]:
        cur_swaps = []
        orig_score = self.lm_scores[s][i][words[i]]
        for swap, score in self.lm_scores[s][i].items():
          if swap == words[i]: continue
          if swap not in self.neighbors[words[i]]: continue
          if score - orig_score >= self.min_log_p_diff:
            cur_swaps.append(swap)
        swaps.append(cur_swaps)
      else:
        swaps.append([])
    return swaps
      
  def check_in(self, words):
    words = [word.lower() for word in words]
    s = ' '.join(words)
    return s in self.lm_scores
