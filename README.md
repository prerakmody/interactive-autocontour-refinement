# Table of Contents

- [Introduction](#1---introduction)
- [Installation](#2---installation)
- [Usage](#3---usage)

# 1 - Introduction
This project is a web-based DICOM viewer and annotation tool. This is specifically used to refine auto-contoured tumor regions from head-andneck scans (CT + PET). 

It contains three components:
1. **Frontend**: Built using node.js and cornerstone3D, this component handles the user interface and interactions.
2. **Backend**: A Python-based FastAPI server that manages the communication between the frontend and the DICOM server (Orthanc).
3. **DICOM Server**: Orthanc is used as the DICOM server to store and retrieve DICOM images.


# 2 - Installation
## 2.1. Using docker

1. Install [docker](https://docs.docker.com/engine/install/).
    - For Linux flavor: `cat /etc/os-release`

2. Install [docker-volume-snapshot](https://github.com/junedkhatri31/docker-volume-snapshot/tree/main)
    - (only for Windows):
      - Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install)
      - Open Ubuntu terminal via Powershell
    - Download [orthanc-config.tar](https://github.com/prerakmody/cornerstone3D-trials/releases/download/v0.2/orthanc-config.tar) and [orthanc-db.tar](https://github.com/prerakmody/cornerstone3D-trials/releases/download/v0.2/orthanc-db.tar) files
    - Then run : 
      - `docker-volume-snapshot restore orthanc-config.tar orthanc-config`
      - `docker-volume-snapshot restore orthanc-db.tar orthanc-db`
    - This will create two volumes: `orthanc-config` and `orthanc-db`.
    - You can check the volumes using the command: `docker volume ls`

3. Copy model files to the `interactive-autocontour-refinement` directory:
    - In [src/backend/_models/](./src/backend/_models/), create a directory named `UNetv1__DICE-LR1e3__W5-B32-MoreData__Cls1-Pt-Scr__Trial1`.
      - Create a subdirectory named `epoch_0090` inside it.
      - Download the model files from [here](https://github.com/prerakmody/cornerstone3D-trials/releases/download/v0.1/epoch_0090)
  
4. Generate `hostCert.pem` and `hostKey.pem`
    - Run `pip install cryptography && python src/backend/utils/certUtils.py`
    - These will automatically be saved to [src/backend/_keys](./src/backend/_keys/) folder.

5. Finally, run the following command to start the server:
    ```bash
    docker-compose up --build
    ```
      - This will build the images and start the containers for the frontend, backend, and DICOM server.

# 3 - Usage
## 3.1 - Local use
1. Go to [http://localhost:50000](http://localhost:50000) in your web browser.
    - For Orthanc db go to [http://localhost:8042](http://localhost:8042)
    - NOTE: Currently, uploading DICOM files into Orthanc might break this repository.

## 3.2 - Remote use
1. For this we will setup nginx as a reverse proxy.
    - Install [nginx](https://www.nginx.com/resources/wiki/start/topics/tutorials/install/)
    - Create a new configuration file in `/etc/nginx/sites-available/interactive-server` and use content from this [file](./src/assets/nginx-interactive-server.conf)
    - Then run the following commands:
      ```bash
      sudo ln -s /etc/nginx/sites-available/interactive-server /etc/nginx/sites-enabled/
      sudo nginx -t
      sudo systemctl restart nginx
      sudo systemctl status nginx.service
      sudo multitail /var/log/nginx/*.log
      ```