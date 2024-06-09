import pickle

with open("encoded_data_charlier_et_al.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
