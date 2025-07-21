# Simple Frontend Dockerfile
FROM nginx:alpine

# Copy static files
COPY public/ /usr/share/nginx/html/
COPY src/ /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 