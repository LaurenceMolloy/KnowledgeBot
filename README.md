data ---- source ----|-- knowledge -- <channel>_YYYYMMDD_<slack timestamp>_<string>
      |-- vectordb   |--

where
YYYYMMDD is the creation date of the message in human-readable format
         (enables easier troubleshooting than just a numeric epoch)
"slack timestamp" is the creation timestamp of the message, with period replaced by an underscore
"string" is first 10 alphanumeric characters of the message block, lower-cased


Building the container
docker build -t knowledgebot . 

Running the container
docker run --rm --name knowledgebot -v "$(pwd)/data:/app/data" knowledgebot

Using docker compose
docker compose up --build