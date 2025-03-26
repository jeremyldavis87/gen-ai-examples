# setup.py
import os
from dotenv import load_dotenv
import argparse
import subprocess

def setup_environment():
    print("Setting up environment for AI Gateway development...")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Creating sample .env file. Please update with your credentials.")
        with open(".env.sample", "r") as sample, open(".env", "w") as env:
            env.write(sample.read())
    
    # Install dependencies
    print("Installing required packages...")
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
    
    print("Environment setup complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup AI Gateway development environment")
    args = parser.parse_args()
    setup_environment()