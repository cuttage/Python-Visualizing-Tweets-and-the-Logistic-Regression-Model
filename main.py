import nltk                         # NLP toolbox
from os import getcwd
import pandas as pd                 # Library for Dataframes
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt     # Library for visualization
import numpy as np                  # Library for math functions

from utils import process_tweet, build_freqs # Our functions for NLP

nltk.download('twitter_samples')
