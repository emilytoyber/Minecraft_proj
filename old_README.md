# Minecraft_proj

Environment checked on python3.8 (pip 20.0.2) and Ubuntu20.04 on WSL2.

## Clone our project
```git clone https://github.com/emilytoyber/Minecraft_proj.git```

## Setting up Environment
```
cd Minecraft_proj/
sudo apt-get install python3.8-venv
python3.8 -m venv {name_of_env}
source {name_of_env}/bin/activate
pip install wheel
pip install -r minerl_req.txt
```

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
```pip install git+https://github.com/minerllabs/minerl #--user```

## Download conda
https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/ (#maybe not needed)