* **To download the artifact**, run:
```
git clone https://github.com/lt-asset/KNOD.git
```
This is enough to obtain all the patches generated KNOD for Defects4J v1.2, Defects4J v2.0 and QuixBugs benchmarks (refers to the result in RQ1 in the paper).


* **To run the code in Docker**, run:
```
docker pull jiang719/knod:latest
docker run -it --name knod --gpus all jiang719/knod:latest
cd /home/KNOD
bash setup.sh
```
More details about running the code can be found in the README.md file.