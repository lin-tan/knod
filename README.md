# KNOD: Domain Knowledge Distilled Tree Decoder for Automated Program Repair
This is the artifact of paper "KNOD: Domain Knowledge Distilled Tree Decoder for Automated Program Repair, Nan Jiang, Thibaud Lutellier, Yiling Lou, Lin Tan, Dan Goldwasser, and Xiangyu Zhang", [ICSE 2023](https://conf.researchr.org/track/icse-2023/icse-2023-technical-track).

## Dependency
* Python 3.10.4
* PyTorch >= 1.9.1
* Java 8
* Docker
* nvidia-docker
* [Defects4J](https://github.com/rjust/defects4j)

## To Run in Docker
* To run this artifact in docker, you need to have [docker](https://docs.docker.com/desktop/install/linux-install/) installed first.
* To use GPUs in docker, you need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
* Pull the docker image, which contains a copy of the whole artifact as well as all the dependencies satisfied: 
```
docker pull jiang719/knod:latest
```
* Start the docker image:
```
docker run -it --name knod --gpus all jiang719/knod:latest
cd /home/KNOD
bash setup.sh
```

## Content
The file structure of the artifact is as follow:
* **data:**
    * **defects4j_input:** model input files of Defects4J benchmark
    * **defects4j_output:** generated patches for Defects4J benchmark
    * **quixbugs_input:** model input files of QuixBugs benchmark
    * **quixbugs_output:** generated patches for QuixBugs benchmark
    * **vocabulary:** vocabulary files for building model
    * **example_training_ast.json:** example training samples
    * **example_validation_ast.json:** example validation samples
    * **jdk.json:** static analysis of JDK-8
* **javalang:** modified [javalang](https://github.com/c2nes/javalang) package by adding API to convert an AST object to source code
* **javaparser:** a Java tool to parse Java files and prepare input for models.
* **results:** KNOD's correct fixes on Defects4J v1.2, Defects4J v2.0 and QuixBugs benchmarks.
* **src:** source code to run KNOD to generate patches or train KNOD models.
* **QuixBugs:** temprary folders for validating QuixBugs bugs.
* **preprint_for_artifact_evaluation.pdf:** An early draft of the preprint for artifact evaluation purpose only (to be updated).

## KNOD's patches
The correct/plausible patches KNOD generated for Defects4J v1.2, Defects4J v2.0 and QuixBugs benchmarks are under the *results* folder, which refers to result of RQ1 in the paper.

## To train KNOD models
```
cd src
python train.py
```
There are several configurations you may want to modify: ```training_data_path``` is the path to the training data, ```validating_data_path``` is the path to the validation data, and ```save_dir``` is the folder to save the models (default set to ```../data/models/```).

Our training data is shared atvia Zenodo at [https://doi.org/10.5281/zenodo.7570475](https://doi.org/10.5281/zenodo.7570475).

## To generate patches for Defects4J
```
cd src
python prepare_defects4j_input.py
python generate_defects4j_output.py
```

## To generate patches for QuixBugs
```
cd src
python prepare_quixbugsj_input.py
python generate_quixbugs_output.py
```

## Citation
If you find this code to be useful for your research, please consider citing:
```
@inproceedings{jiang@domain,
   author = {Jiang, Nan and Lutellier, Thibaud and Lou, Yiling and Tan, Lin and Goldwasser, Dan and Zhang, Xiangyu},
   title = {KNOD: Domain Knowledge Distilled Tree Decoder for Automated Program Repair},
   year = {2023},
   isbn = {9781665457019},
   publisher = {IEEE Press},
   url = {https://doi.org/10.1109/ICSE48619.2023.00111},
   doi = {10.1109/ICSE48619.2023.00111},
   booktitle = {Proceedings of the 45th International Conference on Software Engineering},
   pages = {1251–1263},
   numpages = {13},
   keywords = {abstract syntax tree, deep learning, automated program repair},
   location = {Melbourne, Victoria, Australia},
   series = {ICSE '23}
}
```
