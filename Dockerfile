#Use an official Python runtime as a parent image
FROM python:3.12-slim
#Set the working directory in the container
WORKDIR /app
#Install system dependencies required for WeasyPrint
RUN apt-get update && apt-get install -y build-essential python3-dev pango1.0-tools libpangocairo-1.0-0 --no-install-recommends && rm -rf /var/lib/apt/lists/*
#Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#Copy the rest of the application's code
COPY . .
#Expose the port Gradio runs on
EXPOSE 7860
#Run the Gradio app when the container launches
CMD ["gradio", "app.py", "--server_name", "0.0.0.0"]