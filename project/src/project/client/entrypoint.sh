#!/bin/bash
echo "Waiting for server..."
while ! curl -s http://server:3000 > /dev/null; do
  sleep 1
done
echo "Server ready"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0