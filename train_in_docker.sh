sudo docker run -ti -d --name mlaut-train -v "$(pwd):/mlaut" mlaut:latest /bin/sh -c 'python classic_methods.py'
