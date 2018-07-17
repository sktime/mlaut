sudo docker run -ti -d --name mlaut-train -v "$(pwd):/mlaut" mlaut:latest /bin/sh -c 'python example_train_estimators.py'
