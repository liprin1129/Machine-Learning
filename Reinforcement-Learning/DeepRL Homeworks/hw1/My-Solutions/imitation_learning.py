import numpy as np
import pickle
import os
import argparse

print('working')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)

    args = parser.parse_args()

    with open(args.expert_policy_file, 'rb') as f:
        data = pickle.loads(f.read())
    
    action = policy_fn(obs[None,:])
    print(type(action))

    #print("Observations\n", data["observations"])
    #print("Actions\n", data["actions"])

    
if __name__ == "__main__":
    main()