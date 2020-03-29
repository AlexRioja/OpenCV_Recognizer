# OpenCV_Recognizer

This is a Face Recognizer using OpenCV for Python

### Considerations

TESTED on UBUNTU 19.04

You may have to change some path parameters on Windows Systems!

## USAGE :

Execute the operations in this order:
```
create_datasets.py <dataset_name>
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

## Authors

* **Alejandro Martinez** *


## License

This project is licensed under the MIT License - see [MIT License] (https://opensource.org/licenses/mit-license.php) 
