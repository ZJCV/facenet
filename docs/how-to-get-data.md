
# How to get data

You need to create a data profile for the `facenet` project. 

## File Structure

Assume that the file structure is as follows:

```angular2html
data-root/
    cls_1/
        img_1.jpg
        img_2.jpg
        ...
    cls_2/
        img_1.jpg
        img_2.jpg
        ...
    ...
```

## How to create

```
$ python tools/create_train_test_data.py -h
usage: create_train_test_data.py [-h] [--test] data_root save_root

positional arguments:
  data_root   Data ROOT. Default: None
  save_root   Save ROOT. Default: None

optional arguments:
  -h, --help  show this help message and exit
  --test      Separate training and test set
```

1. Create training and test files separately. 

```angular2html
$ python tools/create_train_test_data.py train_data_root train_save_root
$ python tools/create_train_test_data.py test_data_root test_save_root
```

2. Create training and test files at the same time 

```angular2html
python tools/create_train_test_data.py data_root save_root --test
```

## Related data links

* [LFW](http://vis-www.cs.umass.edu/lfw/index.html#download)