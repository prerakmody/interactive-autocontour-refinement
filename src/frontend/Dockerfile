# Use official Node.js image
FROM node:20

# Set working directory
WORKDIR /app

# Copy the rest of the application
COPY src/frontend/ ./

# Copy package.json and install dependencies
RUN npm install

# Expose port
EXPOSE 50000

# Run
CMD ["sh", "-c", "npx webpack --mode development && npm start"]

# Debug
# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# RUN apt install curl git htop lsof vim tree nginx multitail -y
# curl database:8042/patients
# curl backend:55000/serverHealth
