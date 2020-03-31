# OpenCV_Recognizer

This is a Face Recognizer using OpenCV for Python.
It's thought for cheap systems as it uses Haar like patterns to detect the face and a set of light-weight algorithms (like Fisher, LBHPF or Eiger) to identify the face. 
For that reason, it's not suitable as a security tool.

For a reliable implementation, use OpenCV dnn module (FROM THIS [REPO](https://github.com/AlexRioja/FaceRecognition_OpenCV_DNN))

### Considerations

TESTED on UBUNTU 19.04

You may have to change some path parameters on Windows Systems!

## USAGE :

Execute the operations in this order:
```
create_datasets.py -l <dataset_name> (-c)
```
this will create the dataset in resources/faces_2_recognize. Then execute:

```
train_faces.py
```
this will generate data inside recognizer folder
(YOU WILL NEED AT LEAST 2 DATASETS IN faces_2_recognize FOLDER IN ORDER FOR THIS TO WORK!).
And then enjoy executing:

```
detect_faces.py 
```
## TO-DOs

* Implement algorithm to align faces (the prediction works better if the face is correctly aligned)
* Create Script that works as a Trigger for others retrieving face identification
## Authors

**Alejandro Martinez de Ternero** 


## License

This project is licensed under the MIT License - see [MIT License](https://opensource.org/licenses/mit-license.php) 

