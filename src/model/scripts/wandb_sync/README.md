# Wandb Folder Synchronization Script

## Description

This Python script allows you to synchronize Weights and Biases (wandb) for a specified folder at regular time intervals. It automates the process of running wandb sync on the specified folder at the user-defined time step. This can be useful for keeping your experiment data up-to-date on wandb.

## Prerequisites
Before using this script, make sure you have the following prerequisites installed:
- "wandb>=0.15.2, <0.16"
- "tmux >=3.3, <4"

## how to sync
1. WANDB login
   ```bash
   wandb login [KEY]
   ```

2. Training model with wandb offline mode
   ```bash
   export WANDB_MODE=offline
   ```

3. Run the wandb sync script on tmux
- Start a new session with the name ***_auto_sync_***
   ```bash
   tmux new -s auto_sync
   ```
- Run the script using the following command:
   ```bash
   python sync_wandb.py --foldername FOLDER_NAME --step TIME_STEP
   ```
   - FOLDER_NAME: The name of the folder you want to synchronize with wandb.
   - TIME_STEP: The time step in seconds, which specifies how often you want to sync the folder with wandb. The default is set to 1800 seconds (30 minutes).
- You can detach from session (without cancel your script)
    Ctrl + b and d
- Attach to a session with the name ***_auto_sync_***
   ```bash
   tmux a -t auto_sync
   ```
- Kill/Delete session ***_auto_sync_***
   ```bash
   tmux kill-ses -t auto_sync
   ```

4. View the Results
   Link result will be appear when Run this script
