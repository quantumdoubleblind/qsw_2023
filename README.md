# qsw_2023

## Setup

Clone the repository to a location of your choosing or just download the dockerfile:

```
git clone https://github.com/quantumdoubleblind/qsw_2023.git
```

### Docker

Create a Docker volume to locally store any raw data generated during processing

```
docker volume create 'your-volume-name'
```
Choose a name of your choice to replace **'your-volume-name'**. To confirm that the volume was created successfully, you can use the following command:

```
docker volume ls
```
To locate your volume use the following command:

```
docker volume inspect 'your-volume-name'
```


To create a Docker image, build it using the **'docker build'** command. Once the image has been built, you can run it using the **'docker run'** command. Here are the steps to build and run your Docker image
```
docker build -t 'your-image-name' .
docker run -it -v 'your-volume-name' 'your-image-name'
```
Select a name of your choice to replace **'your-image-name'**. Make sure to run the following commands from the directory that contains your Dockerfile. Once you have started the container with **'docker run'**, you can interactively set various parameters such as graph density, optimization level, number of iterations, and number of parallel threads. Additionally, you can specify the problem for which to generate raw data. To further customize the experiment, you can adjust the parameters in the **'experiments.py'** class.

### Local

```
pip install -r requirements.txt

```



