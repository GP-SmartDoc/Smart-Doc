#!/bin/bash

# This script is intended to be run securely via SSH by a server administrator.
# It triggers a global server wipe, permanently deleting all user documents from the file system 
# and all associated vectors from ChromaDB in the retrieval service.

echo "WARNING: This will permanently delete ALL documents and vector data for ALL users."
read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Initiating global server wipe..."
    
    # Send a POST request to the internal retrieval API's wipe endpoint
    # Adjust the port if your retrieval service runs on a different port internally.
    response=$(curl -s -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/wipe)
    
    if [[ "$response" == *"wiped"* ]]; then
        echo "✅ Server wipe completed successfully."
        echo "Response: $response"
    else
        echo "❌ Failed to wipe the server."
        echo "Response: $response"
        exit 1
    fi
else
    echo "Wipe aborted."
fi
