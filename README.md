# ZHU LI, DO THE THING

1. sudo dockerd &
2. sudo docker build --tag wab665/ascend .
3. sudo docker run -p 8000:8000 wab665/ascend
4. sudo docker push wab665/ascend

# ACR

1. sudo az login
2. sudo az acr login --name ppgcycletime
3. sudo docker tag wab665/ascend ppgcycletime.azurecr.io/wab665/ascend
4. sudo docker push ppgcycletime.azurecr.io/wab665/ascend
5. sudo docker rmi ppgcycletime/wab665/ascend
