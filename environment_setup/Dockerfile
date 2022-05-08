FROM ubuntu:20.04
    MAINTAINER luklun <lukas.lundmark@solita.fi>

# Install Micromamba
RUN apt-get update && apt-get install git wget bzip2 -y && \
    wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest \
    | tar -xvj bin/micromamba --strip-components=1 && \
    cp micromamba /usr/bin/

# Copy the dependency definition to the docker image
COPY ci_dependencies.yml /setup/

# Initialize and activate the environment
RUN micromamba create -f /setup/ci_dependencies.yml -y
ENV PATH /root/micromamba/envs/mlopspython_ci/bin:root/micromamba/condabin:$PATH
RUN az extension add --name ml
