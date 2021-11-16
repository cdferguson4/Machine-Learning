import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

test_games = pd.read_csv("test_esrb.csv")
games = pd.read_csv("gameratings.csv")
