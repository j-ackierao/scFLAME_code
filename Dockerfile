# R version
FROM rocker/r-ver:4.4.1

# System and Python libraries libraries 
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        pkg-config \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        zlib1g-dev libbz2-dev liblzma-dev \
        libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
        libcairo2-dev libxt-dev \
        libfreetype6-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libglpk-dev libgmp-dev \
        libicu-dev \
        libnode-dev \
        pandoc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# R packages (copied before code so this layer is cached)
COPY renv.lock renv.lock
RUN R -e "install.packages(c('renv', 'BiocManager')); renv::restore(library = .libPaths()[1], prompt = FALSE)"

# Setup a python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#Don't let renv renv
ENV RENV_CONFIG_AUTOLOADER_ENABLED=FALSE

COPY . .

CMD ["bash"]