# Minecraft_proj

Environment checked on python3.8 and Ubuntu20.04 on WSL2.

## Downloading Java8
```
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk

# Verify installation
java -version # this should output "1.8.X_XXX"
# If you are still seeing a wrong Java version, you may use
# the following line to update it
# sudo update-alternatives --config java
```

## Download MineRL environment
```pip install git+https://github.com/minerllabs/minerl --user```

## Downgrade numpy because of dependency problems
```pip install --upgrade numpy==1.23.1```

## Clone our project
```git clone https://github.com/emilytoyber/Minecraft_proj.git```
