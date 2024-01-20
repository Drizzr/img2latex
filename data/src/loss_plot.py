#plots the loss of all epochs during training
import matplotlib.pyplot as plt
import os
import argparse
import re

parser = argparse.ArgumentParser(description="Plot the loss of all epochs.")
parser.add_argument("--path", type=str, default="../../checkpoints", help="Path to the checkpoints folder")
args = parser.parse_args()

if not os.path.exists(args.path):
    print("Please provide a valid path to the checkpoints folder (--path <path>)")
    raise Exception("Path does not exist")

if(args.path[-1] == "/"):
    args.path = args.path[:-1]

folders = os.listdir(args.path)

epochs = []
losses = []


for folder in folders:
    pattern = r"chechpoint_epoch_(\d+)_(\d+.\d)+%_estimated_loss_(\d+\.\d+)"
    match = re.match(pattern, folder)

    if match:
        m = int(match.group(1))
        loss = float(match.group(3))
        epochs.append(m)
        losses.append(loss)

epochs, losses = zip(*sorted(zip(epochs, losses)))

print("Best epoch: ", epochs[losses.index(min(losses))])

plt.plot(epochs, losses)
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
