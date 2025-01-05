#!/bin/bash

# Verify if the user is in the root folder
if [ ! -f "download_data.sh" ]; then
  echo "Please run this script from the root folder of the project"
  exit 1
fi

# Download the data from the kaggle
curl -L -o ./140k-real-and-fake-faces.zip\
  https://www.kaggle.com/api/v1/datasets/download/xhlulu/140k-real-and-fake-faces

# Extract to the data folder
mkdir -p data
unzip ./140k-real-and-fake-faces.zip -d data

# Remove the zip file
rm ./140k-real-and-fake-faces.zip

# Remove unexpected 
mv data/real_vs_fake/real-vs-fake/ data/
rm -r data/real_vs_fake

