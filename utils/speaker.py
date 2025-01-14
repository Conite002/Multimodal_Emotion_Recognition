import os
import pandas as pd
import json



def add_speaker_to_json(json_path, dataframe):
    # Load JSON
    with open(json_path, "r") as file:
        data = json.load(file)

    filename_to_speaker = {
        f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4": row['Speaker']
        for _, row in dataframe.iterrows()
    }

    for entry in data:
        filename = os.path.basename(entry['video'])  
        entry['Speaker'] = filename_to_speaker.get(filename, "Unknown")

    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

