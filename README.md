# unet-tf
Unet networks implemented by tensorflow for **Cell Edge Detecting**. The dataset is from [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/). There are only 30 512*512 images, so Data Augmentation has been done by tensorflow API.




## Configure

| configure        | description                                   |
| ---------------- | --------------------------------------------- |
| max_step         | How many steps to train.                      |
| rate             | learning rate for training                    |
| reload_step      | Reload step to continue training              |
| save_interval    | interval to save model                        |
| summary_interval | interval to save summary                      |
| n_classes        | output class number                           |
| batch_size       | batch size for one iteration                  |
| is_training      | training or predict (for batch normalization) |
| datadir          | path to tfrecords                             |
| logdir           | directory to save logs of accuracy and loss   |
| modeldir         | directory to save models                      |
| model_name       | Model name                                    |



## Note

Due to the batch normalization layer, 'is_training=True' should be set during training phase while 'is_training=False' during test phase.



## Dependencies

- **Numpy**
- **PIL**
- **tensorflow (tensorflow-gpu)**
- **os**

These packages are available via **pip install**.



## Run

- #### Data preparation

  The data will be fed into the net in the form of Tfrecords. So first run data.py to generate the tfrecords file. 

  **Note:** the path of tfrecords file should be set in config during training.

  ```python
  if __name__ == '__main__':    
      if not os.path.exists('./data/train/train.tfrecords'):        				
          create_record('data/train', './data/train/train.tfrecords')    		
      else:        
          print('TFRecords already exists!')
  ```

- #### Train

```python
if __name__ == '__main__':   
    model = UNet(tf.Session(), configure())
    model.train()
```

- #### Predict  in test dataset

```python
if __name__ == '__main__':   
    model = UNet(tf.Session(), configure())
    model.predicts()
```



## Results

<img src="https://github.com/lzyhha/unet-tf/raw/master/data/test/26.png"/>

<img src="https://github.com/lzyhha/unet-tf/raw/master/data/predict/26.png"/>

