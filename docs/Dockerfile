FROM pytorch/pytorch:latest

RUN pip3 install --upgrade pip
RUN pip3 install openfermion torch "sympytorch>=0.1.2" rich "tensorboard>=2.12.0" deepdiff jsonschema nevergrad "scipy<1.11" "pyqtorch==0.5.0"

RUN apt update && apt install -y wget unzip fontconfig
RUN wget https://download.jetbrains.com/fonts/JetBrainsMono-2.304.zip
RUN unzip JetBrainsMono-2.304.zip -d JetBrainsMono
RUN mkdir -p /usr/share/fonts/truetype/jetbrains
RUN cp JetBrainsMono/fonts/ttf/*.ttf /usr/share/fonts/truetype/jetbrains/
RUN fc-cache -f -v
