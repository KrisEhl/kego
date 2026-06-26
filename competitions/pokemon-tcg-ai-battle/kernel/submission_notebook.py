"""PTCG AI Battle — Submission Kernel

Dynamically generated to package the agent.
"""

import os
import shutil
import tarfile

INPUT_DIR = "/kaggle/input/competitions/pokemon-tcg-ai-battle"
SAMPLE_DIR = os.path.join(INPUT_DIR, "sample_submission")
WORKING_DIR = "/kaggle/working"

# Write custom main.py
main_py_content = 'import os\nimport random\n\nfrom cg.api import Observation, to_observation_class\n\ndef read_deck_csv() -> list[int]:\n    """Read deck.csv.\n    \n    Returns:\n        list[int]: A list of card IDs in the deck.\n    """\n    file_path = "deck.csv"\n    if not os.path.exists(file_path):\n        file_path = "/kaggle_simulations/agent/" + file_path\n    with open(file_path, "r") as file:\n        csv = file.read().split("\\n")\n    deck = []\n    for i in range(60):\n        deck.append(int(csv[i]))\n    return deck\n\ndef agent(obs_dict: dict) -> list[int]:\n    """Implement Your Pokémon Trading Card Game Agent.\n\n    Each element in the returned list must be >= 0 and < len(obs.select.option).\n    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.\n    \n    Returns:\n        list[int]: A list of option index.\n    """\n    obs: Observation = to_observation_class(obs_dict)\n    if obs.select == None:\n        # In the initial selection, the obs.select is None, and it is necessary to return the deck.\n        # The deck is a list of 60 card IDs.\n        # The deck must comply with the Pokémon Trading Card Game rules.\n        return read_deck_csv()\n    \n    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)  # select randomly\n'
with open(os.path.join(WORKING_DIR, "main.py"), "w") as f:
    f.write(main_py_content)

# Write custom deck.csv
deck_csv_content = "1158\n721\n721\n722\n722\n722\n722\n723\n723\n723\n723\n1145\n1145\n1145\n1145\n1205\n1205\n1227\n1227\n1227\n1227\n1235\n1235\n1235\n1235\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n3\n"
with open(os.path.join(WORKING_DIR, "deck.csv"), "w") as f:
    f.write(deck_csv_content)

# Copy cg directory
cg_src = os.path.join(SAMPLE_DIR, "cg")
cg_dst = os.path.join(WORKING_DIR, "cg")
if os.path.exists(cg_src):
    if os.path.exists(cg_dst):
        shutil.rmtree(cg_dst)
    shutil.copytree(cg_src, cg_dst)

# Create submission.tar.gz
submission_path = os.path.join(WORKING_DIR, "submission.tar.gz")
with tarfile.open(submission_path, "w:gz") as tar:
    for item in ["main.py", "deck.csv", "cg"]:
        full_path = os.path.join(WORKING_DIR, item)
        if os.path.exists(full_path):
            tar.add(full_path, arcname=item)

print(f"Created {submission_path}")
print(f"Size: {os.path.getsize(submission_path)} bytes")

# Verify contents
with tarfile.open(submission_path, "r:gz") as tar:
    print("Contents:")
    for member in tar.getmembers():
        print(f"  {member.name} ({member.size} bytes)")
