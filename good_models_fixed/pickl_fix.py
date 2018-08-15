import pickle
import sys

MODEL_FILE = sys.argv[1]

Q = pickle.load(open("Q_dict_"+MODEL_FILE, "rb"))
accu_rewards = pickle.load(open("train_dict_"+MODEL_FILE, "rb"))

pickle.dump((Q,accu_rewards), open("fixed\\"+MODEL_FILE, "wb"))
