FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application source
COPY . .

# Expose port
EXPOSE 3000

# Run the frontend dev server
CMD ["npx", "vite", "--port=3000", "--host=0.0.0.0"]
